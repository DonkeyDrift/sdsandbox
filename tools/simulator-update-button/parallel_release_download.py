#!/usr/bin/env python3
import argparse
import hashlib
import os
import shutil
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def fmt_size(value):
    if value >= 1024 * 1024 * 1024:
        return f"{value / 1024 / 1024 / 1024:.2f} GiB"
    if value >= 1024 * 1024:
        return f"{value / 1024 / 1024:.2f} MiB"
    return f"{value / 1024:.1f} KiB"


def download_chunk(url, output, start, end, retries=8):
    expected = end - start + 1
    current = output.stat().st_size if output.exists() else 0
    if current > expected:
        output.unlink(missing_ok=True)
        current = 0
    if current == expected:
        return expected

    attempt = 0
    while current < expected:
        attempt += 1
        if attempt > retries:
            raise RuntimeError(f"chunk {start}-{end} failed after {retries} attempts")
        try:
            range_start = start + current
            request = urllib.request.Request(
                url,
                headers={
                    "Range": f"bytes={range_start}-{end}",
                    "User-Agent": "gym-donkeycar-release-downloader/1.0",
                },
            )
            with urllib.request.urlopen(request, timeout=90) as response:
                status = getattr(response, "status", None) or response.getcode() or 200
                if status != 206:
                    raise RuntimeError(f"expected HTTP 206, got {status}")
                with open(output, "ab") as stream:
                    while True:
                        data = response.read(128 * 1024)
                        if not data:
                            break
                        stream.write(data)
                        current += len(data)
                        if current > expected:
                            raise RuntimeError("server returned more chunk bytes than requested")
        except Exception:
            if attempt >= retries:
                raise
            time.sleep(min(2.0 * attempt, 15.0))
    return current


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("output")
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--chunk-size", type=int, default=1024 * 1024)
    args = parser.parse_args()

    output = Path(args.output).absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    work = output.parent / (output.name + ".parts")
    work.mkdir(exist_ok=True)
    chunk_count = (args.size + args.chunk_size - 1) // args.chunk_size
    chunks = []
    for index in range(chunk_count):
        start = index * args.chunk_size
        end = min(start + args.chunk_size, args.size) - 1
        chunks.append((index, start, end, work / f"{index:06d}.part"))

    completed_lock = threading.Lock()
    completed_bytes = 0
    started = time.monotonic()
    failures = []

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(download_chunk, args.url, path, start, end): index
            for index, start, end, path in chunks
        }
        done_count = 0
        for future in as_completed(futures):
            index = futures[future]
            try:
                future.result()
                done_count += 1
                with completed_lock:
                    completed_bytes += chunks[index][2] - chunks[index][1] + 1
                if done_count % 10 == 0 or done_count == chunk_count:
                    elapsed = max(time.monotonic() - started, 0.001)
                    rate = completed_bytes / elapsed
                    print(
                        f"{done_count}/{chunk_count} chunks, {fmt_size(completed_bytes)} done, "
                        f"{fmt_size(rate)}/s",
                        flush=True,
                    )
            except Exception as exc:
                failures.append((index, str(exc)))
                print(f"chunk {index} failed: {exc}", file=sys.stderr, flush=True)

    if failures:
        raise SystemExit(f"{len(failures)} chunk(s) failed; first error: {failures[0][1]}")

    temporary = output.with_suffix(output.suffix + ".tmp")
    hasher = hashlib.sha256()
    with open(temporary, "wb") as destination:
        for _, _, _, path in chunks:
            with open(path, "rb") as source:
                while True:
                    data = source.read(1024 * 1024)
                    if not data:
                        break
                    hasher.update(data)
                    destination.write(data)

    actual_digest = hasher.hexdigest()
    if actual_digest != args.sha256:
        temporary.unlink(missing_ok=True)
        raise SystemExit(f"sha256 mismatch: expected {args.sha256}, got {actual_digest}")
    os.replace(temporary, output)
    shutil.rmtree(work)
    print(f"downloaded {output} ({fmt_size(args.size)}), sha256 {actual_digest}", flush=True)


if __name__ == "__main__":
    main()
