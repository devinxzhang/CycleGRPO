#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import io
import sys
import math
import time
import hashlib
import mimetypes
import threading
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ================== 配置 ==================
INPUT_PARQUET  = "<PATH_TO_DATA>/allenai/pixmo-points/data/train-00000-of-00002.parquet"    # 你的原始 parquet
OUTPUT_PARQUET = "data/pixmo-points/train-00000-of-00002.parquet"  # 输出 parquet
DOWNLOAD_ROOT  = "data/pixmo-points/images"           # 下载目录根路径
MAX_WORKERS    = 32                      # 下载并发数
TIMEOUT        = 30                      # 单次请求超时(s)
RETRIES        = 3                       # 重试次数
SLEEP_BETWEEN  = 0.5                     # 重试间隔(s)
# =========================================

UA = "Mozilla/5.0 (compatible; dataset-downloader/1.0)"
session = requests.Session()
session.headers.update({"User-Agent": UA})

lock = threading.Lock()

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def ext_from_url_or_mime(url: str, content_type: str | None) -> str:
    # 1) 先从URL后缀猜
    parsed = urlparse(url)
    base = os.path.basename(parsed.path)
    if "." in base:
        ext = base.split(".")[-1].lower()
        if ext in ["jpg", "jpeg", "png", "webp", "bmp", "gif", "tiff", "tif"]:
            return "." + ("jpg" if ext == "jpeg" else ext)
    # 2) 从Content-Type猜
    if content_type:
        ext = mimetypes.guess_extension(content_type.split(";")[0].strip())
        if ext:
            # 有些环境会给 .jpe，规范一下
            return ".jpg" if ext == ".jpe" else ext
    # 3) 保底
    return ".jpg"

def sha256_hex(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()

def path_from_sha256(root: str, sha256_str: str, ext: str) -> str:
    # 将图片存储为: root/32/66/3d/.../sha256.ext 或更简单两级
    sub1, sub2 = sha256_str[:2], sha256_str[2:4]
    return os.path.join(root, sub1, sub2, f"{sha256_str}{ext}")

def filename_from_url(root: str, url: str, ext: str) -> str:
    parsed = urlparse(url)
    base = os.path.basename(parsed.path) or "img"
    # 去掉奇怪的query
    base = base.split("?")[0] or "img"
    if "." not in base:
        base = base + ext
    return os.path.join(root, base)

def download_one(row_idx: int, url: str, sha256_expected: str | None):
    """
    下载单个样本:
    - 若提供 sha256，则按 sha256 建路径并校验
    - 否则按 URL 生成文件名
    返回本地文件路径或 None
    """
    if not isinstance(url, str) or not url.startswith(("http://", "https://")):
        return None

    # 预请求 HEAD 获取 content-type（失败也不致命）
    content_type = None
    try:
        r = session.head(url, timeout=TIMEOUT, allow_redirects=True)
        if r.ok:
            content_type = r.headers.get("Content-Type")
    except Exception:
        pass

    # 决定文件扩展名
    ext = ext_from_url_or_mime(url, content_type)

    # 决定本地目标路径
    if isinstance(sha256_expected, str) and len(sha256_expected) == 64:
        out_path = path_from_sha256(DOWNLOAD_ROOT, sha256_expected.lower(), ext)
    else:
        out_path = filename_from_url(DOWNLOAD_ROOT, url, ext)

    ensure_dir(os.path.dirname(out_path))

    # 如果文件已存在，且需要校验，则先校验
    if os.path.exists(out_path):
        if sha256_expected:
            try:
                with open(out_path, "rb") as f:
                    if sha256_hex(f.read()) == sha256_expected.lower():
                        return out_path
            except Exception:
                pass
        else:
            # 无需校验就直接复用
            return out_path

    # 下载 + 重试
    last_err = None
    for attempt in range(1, RETRIES + 1):
        try:
            resp = session.get(url, timeout=TIMEOUT, stream=True)
            if not resp.ok:
                last_err = RuntimeError(f"HTTP {resp.status_code}")
                time.sleep(SLEEP_BETWEEN)
                continue

            # 读取内容到内存（也可以边流式边写文件；为校验方便这里先进内存）
            buf = io.BytesIO()
            for chunk in resp.iter_content(chunk_size=1<<15):
                if chunk:
                    buf.write(chunk)
            data = buf.getvalue()

            # 校验
            if sha256_expected:
                got = sha256_hex(data)
                if got != sha256_expected.lower():
                    last_err = RuntimeError(f"sha256 mismatch: got {got} != {sha256_expected}")
                    time.sleep(SLEEP_BETWEEN)
                    continue

            # 写文件（原子写入）
            tmp = out_path + ".part"
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, out_path)

            return out_path
        except Exception as e:
            last_err = e
            time.sleep(SLEEP_BETWEEN)

    # 失败
    with lock:
        print(f"[WARN] row {row_idx} 下载失败: {url} ({last_err})", file=sys.stderr)
    return None

def main():
    # 读取
    df = pd.read_parquet(INPUT_PARQUET)  # 适合内存能一次装下的情况
    # 若非常大，可考虑 row-group 流式处理，见文末“超大表格流式版本”。

    # 准备字段名
    has_sha = "image_sha256" in df.columns
    if "image_url" not in df.columns:
        raise ValueError("输入 parquet 缺少字段: image_url")

    # 结果列
    image_files = [None] * len(df)

    # 多线程下载
    futures = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for i, row in df.iterrows():
            url = row.get("image_url", None)
            sha = row.get("image_sha256", None) if has_sha else None
            fut = ex.submit(download_one, int(i), url, sha)
            futures[fut] = i

        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                image_files[idx] = fut.result()
            except Exception as e:
                image_files[idx] = None
                with lock:
                    print(f"[ERROR] row {idx} 异常: {e}", file=sys.stderr)

    # 增加新列并写出
    df["image_file"] = image_files

    # 保持原 schema，写 parquet
    table = pa.Table.from_pandas(df, preserve_index=False)
    ensure_dir(os.path.dirname(OUTPUT_PARQUET))
    pq.write_table(table, OUTPUT_PARQUET, compression="zstd")
    print(f"完成: 共 {len(df)} 行；成功 {sum(p is not None for p in image_files)}，输出 {OUTPUT_PARQUET}")

if __name__ == "__main__":
    main()
