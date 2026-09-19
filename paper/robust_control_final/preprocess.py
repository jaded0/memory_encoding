#!/usr/bin/env python3
"""
Preprocess Obsidian-flavored markdown into plain markdown + raw LaTeX
that pandoc can emit cleanly as the body of an IEEE conference paper.

Transformations:
  1. Drop any text before the first '#' heading (e.g. stray author line).
  2. Strip HTML comments ('<!-- ... -->').
  3. Rewrite Obsidian image wikilinks with a following italic caption:
         ![[fig.png]]
         *caption text*
     into standard markdown with alt text used as the figure caption:
         ![caption text](fig.png)
     Fallback: bare ![[fig.png]] -> ![](fig.png)
  4. Strip numeric section numbering from headings (LaTeX auto-numbers).
         '# 1. Introduction' -> '# Introduction'
         '## 6.3.1 LQR'      -> '## LQR'
  5. Transform the '# Abstract\n\n<body>\n\n---' block into a raw
     LaTeX abstract environment, which pandoc passes through verbatim.
  6. Leave everything else (math, blockquotes, [cite …] markers,
     bullet lists) alone — pandoc handles those.

Usage:
    python3 preprocess.py < input.md > processed.md
"""
import re
import sys


def preprocess(text: str) -> str:
    # 1. Drop everything before the first heading line.
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if line.lstrip().startswith("#"):
            text = "\n".join(lines[i:])
            break

    # 2. Strip HTML comments.
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)

    # 3a. Image wikilink immediately followed by an italic caption line.
    #     Capture file name and caption text, emit standard markdown image.
    text = re.sub(
        r"!\[\[([^\]]+)\]\]\s*\n\s*\*([^*\n]+)\*",
        lambda m: f"![{m.group(2).strip()}]({m.group(1).strip()})",
        text,
    )
    # 3b. Any remaining bare wikilink images.
    text = re.sub(r"!\[\[([^\]]+)\]\]", r"![](\1)", text)

    # 4. Strip numeric prefixes from ATX headings. Matches patterns like
    #    '# 1.', '## 2.1', '### 6.3.1.', optionally trailed by a dot.
    text = re.sub(
        r"^(#+)\s+\d+(?:\.\d+)*\.?\s+",
        r"\1 ",
        text,
        flags=re.MULTILINE,
    )

    # 5. Abstract block -> raw LaTeX abstract environment.
    #    Matches the heading, the body up to the next '---' horizontal rule,
    #    and rewrites as \begin{abstract}...\end{abstract}. The raw block
    #    passes through pandoc's markdown reader into LaTeX output verbatim.
    abstract_re = re.compile(
        r"^#\s+Abstract\s*\n+(.*?)\n+---\s*$",
        flags=re.MULTILINE | re.DOTALL,
    )

    def abstract_sub(m: re.Match) -> str:
        body = m.group(1).strip()
        return f"\\begin{{abstract}}\n{body}\n\\end{{abstract}}\n"

    text = abstract_re.sub(abstract_sub, text, count=1)

    # 6. Drop the trailing skeleton sections (manual References stub, the
    #    "to ai" note, and the AI editing-notes block). A real bibliography
    #    is added by the shell .tex via \bibliography{robust_control}.
    text = re.split(r"^#\s+References\s*$", text, maxsplit=1, flags=re.MULTILINE)[0]
    text = text.rstrip().rstrip("-").rstrip()  # drop the trailing '---' rule

    # 7. Convert '[cite key1, key2]' markers into '\cite{key1,key2}'.
    #    The non-key marker 'course readings' maps to the course-notes entry.
    KEY_ALIAS = {"course readings": "Dahleh6241J"}

    def cite_sub(m: re.Match) -> str:
        raw = m.group(1)
        keys = [KEY_ALIAS.get(k.strip(), k.strip()) for k in raw.split(",")]
        return "\\cite{" + ",".join(keys) + "}"

    text = re.sub(r"\[cite\s+([^\]]+)\]", cite_sub, text)

    return text


if __name__ == "__main__":
    sys.stdout.write(preprocess(sys.stdin.read()))
