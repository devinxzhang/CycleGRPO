import json
import os
from pathlib import Path
import subprocess

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from generalized.preflight import check_checkpoint, check_data


def checkpoint(path):
    path.mkdir()
    (path / 'config.json').write_text('{}')
    (path / 'tokenizer_config.json').write_text('{}')
    (path / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'weight': 'model.safetensors'}}))
    (path / 'model.safetensors').write_bytes(b'fixture-only')
    return path


def data(path, sources):
    rows = [dict(source=source, videos=['fixture.mp4'], cap_problem='<video> Describe 1-2 seconds.',
                 cap_answer='', seg_problem='Localize.', seg_answer='<time>1-2 seconds</time>',
                 sampling_weight=1.0) for source in sources]
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


def test_incomplete_checkpoint_fails(tmp_path):
    model = checkpoint(tmp_path / 'model')
    assert check_checkpoint(model) == 1
    (model / 'model.safetensors').unlink()
    with pytest.raises(ValueError, match='Missing or empty'):
        check_checkpoint(model)


def test_main_mix_cannot_be_mislabeled_as_temporal_only(tmp_path):
    parquet = data(tmp_path / 'train.parquet', ['stgr_cyclegrpo'])
    with pytest.raises(ValueError, match='Expected temporal'):
        check_data(parquet, 'temporal_qwen')


@pytest.mark.parametrize('profile', ['temporal_qwen', 'temporal_omtg', 'openo3_mixed'])
def test_public_launcher_dry_run(tmp_path, profile):
    root = Path(__file__).resolve().parents[2]
    model = checkpoint(tmp_path / 'model')
    sources = ['tg_multi_merged'] if profile != 'openo3_mixed' else [
        'dam_cyclegrpo', 'tg_multi_merged', 'stgr_cyclegrpo', 'open_o3_understanding_grpo']
    parquet = data(tmp_path / 'train.parquet', sources)
    env = dict(os.environ, MODEL_PATH=str(model), TRAIN_FILE=str(parquet), DRY_RUN='1')
    result = subprocess.run(['bash', 'generalized/train.sh', profile], cwd=root,
                            env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert 'Configuration resolved; not starting Ray.' in result.stdout
    assert f"model_path: {model}" in result.stdout
