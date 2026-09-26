"""Create a deterministic, small I2T Parquet sample without loading the full dataset."""

import argparse
import math
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def prepare_i2t_subset(input_path, output_path, max_samples=10_000, seed=42):
    source = Path(input_path).resolve()
    destination = Path(output_path).resolve()
    if source == destination:
        raise ValueError("input and output paths must be different")
    if max_samples < 1:
        raise ValueError("max_samples must be positive")

    parquet = pq.ParquetFile(source)
    required = {"conversations", "image_bytes"}
    if not required.issubset(parquet.schema_arrow.names):
        raise ValueError(f"I2T parquet must contain columns: {sorted(required)}")

    row_counts = [parquet.metadata.row_group(i).num_rows for i in range(parquet.num_row_groups)]
    total_rows = sum(row_counts)
    sample_count = min(max_samples, total_rows)
    if total_rows == 0:
        raise ValueError("input parquet is empty")

    median_group_rows = sorted(row_counts)[len(row_counts) // 2]
    group_count = min(
        len(row_counts),
        max(1, 16, math.ceil(sample_count * 20 / median_group_rows)),
    )
    rng = random.Random(seed)
    offset = rng.randrange(len(row_counts))
    group_ids = sorted({
        (offset + i * len(row_counts) // group_count) % len(row_counts)
        for i in range(group_count)
    })
    pool_size = sum(row_counts[i] for i in group_ids)
    quotas = [sample_count * row_counts[i] // pool_size for i in group_ids]
    remainder = sample_count - sum(quotas)
    order = sorted(
        range(len(group_ids)),
        key=lambda j: (-(sample_count * row_counts[group_ids[j]] % pool_size), j),
    )
    for j in order[:remainder]:
        quotas[j] += 1

    samples = []
    for group_id, quota in zip(group_ids, quotas):
        if quota == 0:
            continue
        group = parquet.read_row_group(group_id)
        selected = rng.sample(range(row_counts[group_id]), quota)
        samples.append(group.take(pa.array(selected, type=pa.int64())))

    table = pa.concat_tables(samples)
    table = table.take(pa.array(rng.sample(range(sample_count), sample_count), type=pa.int64()))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        pq.write_table(table, temporary, compression="zstd")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return total_rows, sample_count, len(group_ids)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="dataset/sft_i2t.parquet")
    parser.add_argument("--output", default="dataset/sft_i2t_mini.parquet")
    parser.add_argument("--max_samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    total, sampled, groups = prepare_i2t_subset(args.input, args.output, args.max_samples, args.seed)
    print(f"Wrote {sampled:,} of {total:,} rows using {groups} spaced row groups to {args.output}")


if __name__ == "__main__":
    main()
