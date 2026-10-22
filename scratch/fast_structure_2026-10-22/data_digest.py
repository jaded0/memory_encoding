"""md5 of the text column of each split of the given synth_datasets (to check that two machines'
copies hold the same rows in the same order). Usage: python data_digest.py synth_datasets/NAME ..."""
import hashlib
import sys

from datasets import load_from_disk

for path in sys.argv[1:]:
    data = load_from_disk(path)
    for split in sorted(data):
        digest = hashlib.md5("\n".join(data[split]["text"]).encode()).hexdigest()
        print(f"{path.rstrip('/').split('/')[-1]} {split} {len(data[split])} {digest}")
