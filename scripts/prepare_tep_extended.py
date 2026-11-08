"""Verify and repack the downloaded TEP extended Parquet mirror by run.

The source Parquet files remain untouched.  Output files contain one complete
simulation run per row group, enabling bounded-memory random window access.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "fault_free_train.parquet": (250_000, 500, "08ef1e10e428f5df205d3fc7393ddc28ea9b1d52b223f0c20f397d2fb96394e3"),
    "fault_free_test.parquet": (480_000, 960, "12068eb52dbaeed8ccb206f64aca0aa0f0b65d2bcdbb6e6512b49eb3760aa6d8"),
    "faulty_train.parquet": (5_000_000, 500, "68e7d14c5217e893675b5522c863404e6b2d735fa5b2ab1c15970684b4953ce1"),
    "faulty_test.parquet": (9_600_000, 960, "04b64957e89f6bc98ecb199fd18bf4312549e6909b937383e083684919cac970"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def output_is_valid(path: Path, rows: int, run_length: int) -> bool:
    if not path.exists():
        return False
    parquet = pq.ParquetFile(path)
    expected_groups = rows // run_length
    if parquet.metadata.num_rows != rows or parquet.metadata.num_row_groups != expected_groups:
        return False
    return all(
        parquet.metadata.row_group(index).num_rows == run_length
        for index in range(expected_groups)
    )


def repack(source: Path, destination: Path, rows: int, run_length: int, force: bool) -> None:
    if output_is_valid(destination, rows, run_length) and not force:
        print(f"[skip] {destination.name}: existing run-indexed file is valid")
        return
    source_file = pq.ParquetFile(source)
    if source_file.metadata.num_rows != rows:
        raise ValueError(f"{source}: expected {rows} rows, found {source_file.metadata.num_rows}")
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    writer = pq.ParquetWriter(temporary, source_file.schema_arrow, compression="zstd")
    count = 0
    try:
        for batch in source_file.iter_batches(batch_size=run_length):
            if batch.num_rows != run_length:
                raise ValueError(
                    f"{source}: final batch has {batch.num_rows} rows; expected complete runs of {run_length}"
                )
            writer.write_batch(batch, row_group_size=run_length)
            count += batch.num_rows
            if count % (run_length * 1000) == 0:
                print(f"  {source.name}: {count:,}/{rows:,} rows")
    finally:
        writer.close()
    if count != rows:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"{source}: wrote {count} rows, expected {rows}")
    temporary.replace(destination)
    if not output_is_valid(destination, rows, run_length):
        raise RuntimeError(f"Output validation failed: {destination}")
    print(f"[pass] {destination.name}: {rows // run_length:,} run row groups")


def main() -> None:
    default_root = PROJECT_ROOT / "datasets" / "tep" / "extended500"
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=default_root)
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    source_dir = arguments.root.expanduser().resolve() / "parquet"
    output_dir = arguments.root.expanduser().resolve() / "run_parquet"
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, (rows, run_length, expected_hash) in FILES.items():
        source = source_dir / name
        if not source.exists():
            raise FileNotFoundError(source)
        actual_hash = sha256(source)
        if actual_hash != expected_hash:
            raise ValueError(f"SHA-256 mismatch for {source}: {actual_hash}")
        print(f"[hash pass] {name}")
        repack(source, output_dir / name, rows, run_length, arguments.force)


if __name__ == "__main__":
    main()


# Design note (paper Sec. 4.1): row groups are physical run containers only.
# Runs remain independent units for splitting; window extraction happens later.
