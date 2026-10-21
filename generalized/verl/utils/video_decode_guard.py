"""Killable guard for native video decoders used by CycleGRPO.

Decord/FFmpeg can block forever in native code on a damaged H.264 stream.  A
Python thread timeout cannot interrupt that call, so decoding is delegated to
a persistent spawned process.  If one request exceeds its deadline, the child
is terminated and recreated; the caller receives an exception containing the
exact media path instead of leaving every distributed rank waiting forever.
"""

from __future__ import annotations

import atexit
import multiprocessing as mp
import os
import pickle
import shutil
import tempfile
import threading
import time
import traceback
from typing import Any


class VideoDecodeTimeout(RuntimeError):
    """A native video decode exceeded its configured deadline."""


class VideoDecodeError(RuntimeError):
    """A guarded video decode failed in the helper process."""


_TENSOR_WIRE_KEY = "__cyclegrpo_tensor_numpy__"
_TUPLE_WIRE_KEY = "__cyclegrpo_tuple__"


def _to_wire(value: Any) -> Any:
    """Serialize tensors without torch's privileged shared-memory sockets."""
    # Import lazily so the controller can start and run timeout-only tests
    # without initializing the full torch runtime in the helper.
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None and isinstance(value, torch.Tensor):
        return {_TENSOR_WIRE_KEY: value.detach().cpu().numpy()}
    if isinstance(value, dict):
        return {key: _to_wire(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return {_TUPLE_WIRE_KEY: [_to_wire(item) for item in value]}
    if isinstance(value, list):
        return [_to_wire(item) for item in value]
    return value


def _from_wire(value: Any) -> Any:
    if isinstance(value, dict) and set(value) == {_TENSOR_WIRE_KEY}:
        import torch

        return torch.from_numpy(value[_TENSOR_WIRE_KEY])
    if isinstance(value, dict) and set(value) == {_TUPLE_WIRE_KEY}:
        return tuple(_from_wire(item) for item in value[_TUPLE_WIRE_KEY])
    if isinstance(value, dict):
        return {key: _from_wire(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_from_wire(item) for item in value]
    return value


def _decode_worker(connection) -> None:
    # Decoder helpers are CPU-only.  Prevent a helper spawned by a GPU Ray
    # actor from creating its own CUDA context and pinning extra HBM.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_FLAX", "0")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/cyclegrpo-matplotlib")
    # Keep Decord's internal pool small.  Sixteen Ray ranks decoding at once
    # otherwise create hundreds of FFmpeg threads and amplify stragglers.
    os.environ.setdefault(
        "CYCLEGRPO_DECORD_NUM_THREADS",
        os.environ.get("CYCLEGRPO_DECORD_NUM_THREADS", "2"),
    )
    result_dir = tempfile.mkdtemp(prefix="cyclegrpo-video-decoder-")
    connection.send(("ready", result_dir))
    operations = None
    while True:
        request = connection.recv()
        if request is None:
            return
        request_id, operation, args, kwargs = request
        try:
            if operation == "_test_sleep":
                time.sleep(float(args[0]))
                result = None
            else:
                if operations is None:
                    import decord

                    original_video_reader = decord.VideoReader
                    decoder_threads = int(
                        os.environ.get("CYCLEGRPO_DECORD_NUM_THREADS", "2")
                    )

                    def bounded_video_reader(*reader_args, **reader_kwargs):
                        reader_kwargs.setdefault("num_threads", decoder_threads)
                        return original_video_reader(*reader_args, **reader_kwargs)

                    # qwen-vl-utils constructs decord.VideoReader without a
                    # num_threads argument.  Patch only this isolated helper
                    # process so every route, including fetch_video(), obeys
                    # the same bounded decoder pool.
                    decord.VideoReader = bounded_video_reader
                    from . import dataset as dataset_module

                    operations = {
                        "video": dataset_module.process_video,
                        "open_o3_frames": dataset_module.process_open_o3_frame_sequence,
                        "atomic_query": dataset_module.process_atomic_spatiotemporal_query_video,
                        "trajectory_query": dataset_module.process_trajectory_spatiotemporal_query_video,
                        "joint_query": dataset_module.process_joint_spatiotemporal_query_video,
                    }
                result = operations[operation](*args, **kwargs)
            # Sending decoded frames directly through multiprocessing.Pipe can
            # deadlock after poll() reports the first readable bytes: recv()
            # then waits without a deadline for the rest of a large pickle.
            # Materialize the payload first and send only a small file path.
            result_path = os.path.join(result_dir, f"result-{request_id}.pkl")
            temporary_path = result_path + ".tmp"
            with open(temporary_path, "wb") as result_file:
                pickle.dump(
                    _to_wire(result),
                    result_file,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            os.replace(temporary_path, result_path)
            connection.send((request_id, True, result_path))
        except BaseException as exc:  # propagate decoder crashes with context
            connection.send(
                (
                    request_id,
                    False,
                    (type(exc).__name__, str(exc), traceback.format_exc()),
                )
            )


class _DecodeGuard:
    def __init__(self) -> None:
        self._context = mp.get_context("spawn")
        self._process = None
        self._connection = None
        self._request_id = 0
        self._lock = threading.Lock()
        self._result_dir = None

    def _stop(self) -> None:
        process, connection, result_dir = (
            self._process,
            self._connection,
            self._result_dir,
        )
        self._process = None
        self._connection = None
        self._result_dir = None
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass
        if process is not None and process.is_alive():
            process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
        if result_dir is not None:
            shutil.rmtree(result_dir, ignore_errors=True)

    def _start(self, startup_timeout_s: float) -> None:
        self._stop()
        parent, child = self._context.Pipe(duplex=True)
        process = self._context.Process(
            target=_decode_worker,
            args=(child,),
            name="cyclegrpo-video-decoder",
            daemon=True,
        )
        process.start()
        child.close()
        self._process = process
        self._connection = parent
        if not parent.poll(startup_timeout_s):
            self._stop()
            raise VideoDecodeTimeout(
                f"video decoder helper did not start within {startup_timeout_s:.1f}s"
            )
        status, result_dir = parent.recv()
        if status != "ready":
            self._stop()
            raise VideoDecodeError("video decoder helper returned an invalid handshake")
        self._result_dir = result_dir

    def call(
        self,
        operation: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        *,
        timeout_s: float,
        media_path: str,
        context: str,
    ) -> Any:
        with self._lock:
            startup_timeout_s = max(30.0, timeout_s)
            if self._process is None or not self._process.is_alive():
                self._start(startup_timeout_s)
            self._request_id += 1
            request_id = self._request_id
            started = time.monotonic()
            try:
                self._connection.send((request_id, operation, args, kwargs))
            except (BrokenPipeError, EOFError, OSError) as exc:
                self._stop()
                raise VideoDecodeError(
                    f"video decoder helper died before request; context={context}; "
                    f"operation={operation}; media={media_path}: {exc}"
                ) from exc
            response_holder = []
            response_error = []
            response_ready = threading.Event()

            def receive_response() -> None:
                try:
                    response_holder.append(self._connection.recv())
                except BaseException as exc:
                    response_error.append(exc)
                finally:
                    response_ready.set()

            receiver = threading.Thread(
                target=receive_response,
                name="cyclegrpo-video-decoder-recv",
                daemon=True,
            )
            receiver.start()
            if not response_ready.wait(timeout_s):
                self._stop()
                raise VideoDecodeTimeout(
                    f"video decode timed out after {timeout_s:.1f}s; "
                    f"context={context}; operation={operation}; media={media_path}"
                )
            if response_error:
                exc = response_error[0]
                self._stop()
                raise VideoDecodeError(
                    f"video decoder helper died during request; context={context}; "
                    f"operation={operation}; media={media_path}: {exc}"
                ) from exc
            response_id, ok, payload = response_holder[0]
            elapsed = time.monotonic() - started
            if response_id != request_id:
                self._stop()
                raise VideoDecodeError(
                    f"video decoder response id mismatch; context={context}; "
                    f"media={media_path}"
                )
            if not ok:
                error_type, message, child_traceback = payload
                raise VideoDecodeError(
                    f"video decode failed after {elapsed:.2f}s; context={context}; "
                    f"operation={operation}; media={media_path}; "
                    f"error={error_type}: {message}\n{child_traceback}"
                )
            try:
                with open(payload, "rb") as result_file:
                    result = pickle.load(result_file)
            finally:
                try:
                    os.unlink(payload)
                except FileNotFoundError:
                    pass
            return _from_wire(result)


_GUARD = _DecodeGuard()
atexit.register(_GUARD._stop)


def guarded_video_decode(
    operation: str,
    *args: Any,
    timeout_s: float | None = None,
    media_path: str | None = None,
    context: str = "unknown",
    **kwargs: Any,
) -> Any:
    """Run a supported decode operation with a killable native-code timeout."""

    # A validated offline frame cache cannot block in FFmpeg.  Resolve it in
    # the caller process so a large frame tensor is not serialized through the
    # helper's pickle file only to return to this same process.
    if operation == "video":
        from .video_frame_cache import load_cache_entry

        video = args[0]
        min_pixels = args[1]
        max_pixels = args[2]
        video_fps = args[3]
        cached = load_cache_entry(
            video,
            min_pixels,
            max_pixels,
            video_fps,
            requested_nframes=kwargs.get("nframes"),
        )
        if cached is not None:
            frames, video_metadata, sample_fps = cached
            if kwargs.get("return_video_metadata", False):
                video_metadata = dict(video_metadata)
                if kwargs.get("use_sample_fps_metadata", False):
                    video_metadata["fps"] = float(sample_fps)
                return frames, video_metadata
            if kwargs.get("return_fps", False):
                return frames, sample_fps
            return frames

    if timeout_s is None:
        timeout_s = float(os.environ.get("CYCLEGRPO_VIDEO_DECODE_TIMEOUT_S", "60"))
    if timeout_s <= 0:
        # Primarily useful for controlled A/B debugging.
        from . import dataset as dataset_module

        direct = {
            "video": dataset_module.process_video,
            "open_o3_frames": dataset_module.process_open_o3_frame_sequence,
            "atomic_query": dataset_module.process_atomic_spatiotemporal_query_video,
            "trajectory_query": dataset_module.process_trajectory_spatiotemporal_query_video,
            "joint_query": dataset_module.process_joint_spatiotemporal_query_video,
        }
        return direct[operation](*args, **kwargs)
    path = str(media_path if media_path is not None else (args[0] if args else "<unknown>"))
    return _GUARD.call(
        operation,
        tuple(args),
        kwargs,
        timeout_s=float(timeout_s),
        media_path=path,
        context=context,
    )
