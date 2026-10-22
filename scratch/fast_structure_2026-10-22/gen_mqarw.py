"""Generate the wide-key MQAR sets for the mask load test: the training mixture at full size,
the fixed-load evaluation sets with a small train split (only validation/test are read)."""
import sys
import bench_tasks
out = sys.argv[1] if len(sys.argv) > 1 else "synth_datasets"
bench_tasks.generate_dataset("mqarw_r1_52_q8", out_dir=out)
for k in bench_tasks.WIDE_LOADS:
    bench_tasks.generate_dataset(f"mqarw_{k}_q{min(k, 8)}", out_dir=out,
                                 sizes={"train": 2000, "validation": 5000, "test": 20000})
