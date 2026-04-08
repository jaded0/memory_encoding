#!/usr/bin/env python3
"""
Post-process the pandoc-generated body to fix up constructs that don't
work with the ieeeconf (two-column IEEE conference) class.

Transformation: rewrite pandoc's longtable blocks into a plain `tabular`
wrapped in a `table` environment. `longtable` errors out in two-column
mode, so we parse out the header row and body rows and emit a tabular
with a simple column spec sized for the 88mm IEEE column.

Usage:
    python3 postprocess.py < paper_content.tex > paper_content.tex.new
"""
import re
import sys


def convert_longtable(text: str) -> str:
    """Find each \\begin{longtable}...\\end{longtable} block, extract the
    header row and body rows, and rebuild as a simple tabular."""
    out = []
    i = 0
    while True:
        start = text.find("\\begin{longtable}", i)
        if start == -1:
            out.append(text[i:])
            break
        out.append(text[i:start])
        end = text.find("\\end{longtable}", start)
        if end == -1:
            # Unmatched — give up on this block, emit as-is.
            out.append(text[start:])
            break
        block = text[start : end + len("\\end{longtable}")]

        # --- Header row: between \toprule and \midrule ---
        header_match = re.search(
            r"\\toprule(?:\\noalign\{\})?\s*\n(.*?)\\midrule",
            block,
            flags=re.DOTALL,
        )
        header_rows = header_match.group(1).strip() if header_match else ""

        # --- Body rows: between \endlastfoot (or \endhead if no foot) and
        #     \end{longtable} ---
        body_match = re.search(
            r"\\endlastfoot\s*\n(.*?)\\end\{longtable\}",
            block,
            flags=re.DOTALL,
        )
        if not body_match:
            body_match = re.search(
                r"\\endhead\s*\n(.*?)\\end\{longtable\}",
                block,
                flags=re.DOTALL,
            )
        body_rows = body_match.group(1).strip() if body_match else ""

        # Unwrap the \begin{minipage}[b]{\linewidth}\raggedright ... \end{minipage}
        # wrappers pandoc adds around header cells.
        def strip_minipages(s: str) -> str:
            s = re.sub(
                r"\\begin\{minipage\}\[[bt]\]\{\\linewidth\}\\raggedright\s*",
                "",
                s,
            )
            s = re.sub(r"\s*\\end\{minipage\}", "", s)
            return s

        header_rows = strip_minipages(header_rows)
        body_rows = strip_minipages(body_rows)

        # Drop any stray \bottomrule that was sitting in the longtable
        # foot — we'll re-add it at the end of the body.
        body_rows = re.sub(r"\\bottomrule(?:\\noalign\{\})?\s*", "", body_rows)
        body_rows = body_rows.rstrip()

        # Narrow two-column spec that fits the 88 mm IEEE column.
        colspec = (
            r"@{}>{\raggedright\arraybackslash}p{0.22\columnwidth}"
            r">{\raggedright\arraybackslash}p{0.68\columnwidth}@{}"
        )

        rebuilt = (
            "\\begin{table}[h]\n"
            "\\centering\n"
            "\\footnotesize\n"
            f"\\begin{{tabular}}{{{colspec}}}\n"
            "\\toprule\n"
            f"{header_rows}\n"
            "\\midrule\n"
            f"{body_rows}\n"
            "\\bottomrule\n"
            "\\end{tabular}\n"
            "\\end{table}"
        )
        out.append(rebuilt)
        i = end + len("\\end{longtable}")

    return "".join(out)


def fix_escaped_math_in_tables(text: str) -> str:
    """Pandoc escapes dollar signs inside pipe-table cells instead of
    parsing them as math delimiters, so `$\\sim$` becomes `\\$\\sim\\$`
    (literal dollars with a stray \\sim between them), which blows up
    pdflatex because \\sim is undefined in text mode. Re-math-ify any
    paired \\$...\\$ whose content begins with a backslash (i.e. is a
    math command) so the original intent survives."""
    return re.sub(r"\\\$(\\[^$]*?)\\\$", r"$\1$", text)


def postprocess(text: str) -> str:
    text = convert_longtable(text)
    text = fix_escaped_math_in_tables(text)
    return text


if __name__ == "__main__":
    sys.stdout.write(postprocess(sys.stdin.read()))
