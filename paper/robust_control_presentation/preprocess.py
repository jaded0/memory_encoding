#!/usr/bin/env python3
"""
Preprocess Obsidian-flavored markdown into plain markdown that pandoc can
emit cleanly as the body of a beamer presentation.

Transformations:
  1. Drop any leading Obsidian tag-only line (e.g. '#hebby #phd').
  2. Drop the title block: everything from the first ATX heading down to
     and including the first '---' horizontal rule. The title slide is
     produced by the shell .tex via \\title / \\subtitle / \\maketitle.
  3. Strip HTML comments ('<!-- ... -->').
  4. Rewrite Obsidian image wikilinks to standard markdown:
         ![[file.png]] -> ![](file.png)
     and rewrite 'Pasted image YYYYMMDDHHMMSS.png' filenames to a sanitized
     spaces-free form ('pasted_YYYYMMDDHHMMSS.png') so the symlinks under
     this directory resolve cleanly inside \\includegraphics.
  5. Promote any single-hash heading ('# Foo') to a double-hash heading
     ('## Foo') so that pandoc with --slide-level=2 treats every heading
     as a slide title (the source mixes the two levels for slides).
  6. Leave everything else (math, blockquotes, bullet lists, tables) alone.

Usage:
    python3 preprocess.py < input.md > processed.md
"""
import re
import sys


PASTED_RE = re.compile(r"Pasted image (\d+)\.png", re.IGNORECASE)


def sanitize_pasted(name: str) -> str:
    """'Pasted image 20251030104039.png' -> 'pasted_20251030104039.png'."""
    return PASTED_RE.sub(lambda m: f"pasted_{m.group(1)}.png", name)


def preprocess(text: str) -> str:
    lines = text.split("\n")

    # 1. Drop a leading tag-only line. Obsidian tags look like '#hebby #phd'
    #    (no space after the hash, multiple tags on one line). A real ATX
    #    heading requires a space after the hash, so 'tag-only' is detected
    #    as: starts with '#', no space immediately after, no space-prefixed
    #    word in the line.
    while lines and lines[0].strip() == "":
        lines.pop(0)
    if lines and re.match(r"^\s*#\S", lines[0]) and not re.match(r"^\s*#+\s", lines[0]):
        lines.pop(0)
    text = "\n".join(lines)

    # 2. Drop the title block: from the first ATX heading down to and
    #    including the first '---' horizontal rule on its own line. The
    #    title slide is rendered by the shell .tex.
    title_block_re = re.compile(
        r"^#+\s.*?\n---\s*$",
        flags=re.DOTALL | re.MULTILINE,
    )
    text = title_block_re.sub("", text, count=1).lstrip()

    # 3. Strip HTML comments.
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)

    # 4a. Sanitize 'Pasted image …' filenames everywhere they appear (mostly
    #     inside wikilink image references, but be defensive).
    text = PASTED_RE.sub(lambda m: f"pasted_{m.group(1)}.png", text)

    # 4b. Wikilink images with optional italic caption.
    text = re.sub(
        r"!\[\[([^\]]+)\]\]\s*\n\s*\*([^*\n]+)\*",
        lambda m: f"![{m.group(2).strip()}]({m.group(1).strip()})",
        text,
    )
    text = re.sub(r"!\[\[([^\]]+)\]\]", r"![](\1)", text)

    # 5. Promote any '# heading' to '## heading' so --slide-level=2 covers
    #    every slide title uniformly. The source mixes '# Foo' and '## Foo'
    #    for what are conceptually all top-level slides; without this step,
    #    the '# Foo' ones would render as section dividers and orphan their
    #    slide bodies.
    text = re.sub(r"^# (?!#)", "## ", text, flags=re.MULTILINE)

    return text


if __name__ == "__main__":
    sys.stdout.write(preprocess(sys.stdin.read()))
