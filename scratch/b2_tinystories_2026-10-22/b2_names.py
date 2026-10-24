"""B2 (2026-10-22): the name lexicon for re-mention surprisal, from the first 100k TinyStories train stories.

A name is a word (maximal [A-Za-z] run) whose first letter is upper case and the rest lower case, length >= 2,
with >= MIN_MID mid-sentence occurrences (preceded by a lowercase letter or a comma, then one space) of which
>= 95% are capitalized (counting the lowercase form's mid-sentence occurrences too), minus a stoplist of titles
and family words. Pre-registered in PREREG_fastlayers.md, Batch B2, measure (b).

    python b2_names.py PROCESSED_TRAIN_DIR OUT.json
"""
import collections
import json
import re
import sys

from datasets import load_from_disk

STOP = {"I", "Mom", "Mommy", "Dad", "Daddy", "Mum", "Mama", "Papa", "Grandma", "Grandpa", "Granny", "Mr", "Mrs",
        "Miss", "Ms", "Sir", "Dr", "God"}
MIN_MID, MIN_CAP_SHARE, STORIES = 20, 0.95, 100_000
WORD = re.compile(r"[A-Za-z]+")


def mid_sentence(text, start):
    """True if the word at `start` follows '<lowercase letter or comma> '."""
    return start >= 2 and text[start - 1] == " " and (text[start - 2].islower() or text[start - 2] == ",")


def main(train_dir, out_path):
    texts = load_from_disk(train_dir).select(range(STORIES))["text"]
    cap_mid, low_mid, total = collections.Counter(), collections.Counter(), collections.Counter()
    for text in texts:
        for match in WORD.finditer(text):
            word = match.group()
            total[word] += 1
            if not mid_sentence(text, match.start()):
                continue
            if word[0].isupper() and (len(word) == 1 or word[1:].islower()):
                cap_mid[word] += 1
            elif word.islower():
                low_mid[word.capitalize()] += 1
    names = {}
    for word, count in cap_mid.items():
        if len(word) < 2 or word in STOP or count < MIN_MID:
            continue
        share = count / (count + low_mid[word])
        if share >= MIN_CAP_SHARE:
            names[word] = {"mid_capitalized": count, "mid_lowercase": low_mid[word], "share": round(share, 4),
                           "occurrences": total[word]}
    ranked = dict(sorted(names.items(), key=lambda item: -item[1]["occurrences"]))
    with open(out_path, "w") as handle:
        json.dump({"stories": STORIES, "min_mid": MIN_MID, "min_cap_share": MIN_CAP_SHARE, "stoplist": sorted(STOP),
                   "names": ranked}, handle, indent=1)
    print(len(ranked), "names; most frequent:", list(ranked)[:40])


if __name__ == "__main__":
    main(*sys.argv[1:])
