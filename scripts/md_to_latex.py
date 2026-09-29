#!/usr/bin/env python3
"""Convert THESIS/markdown/*.md thesis chapters to LaTeX and compile main.pdf.

Follows the author's conversion rules documented inside the markdown itself:
  - #todo / #idea / #nota / #note / ... tags are NOT exported;
  - the gray ``<span ...>[label]</span>`` scaffold labels are NOT exported;
  - "Note for AI" blockquotes are NOT exported;
  - ~~strikethrough~~ text is NOT exported;
  - [[wiki-links]] and --- rules are dropped;
  - square-bracket placeholders ([dataset_size], [W], ...) are replaced with the
    current values from _ai-info_.md.

Outputs THESIS/latex/main.tex + chapters/*.tex, then compiles main.pdf with
tectonic into the repository root.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MD = REPO / "THESIS" / "markdown"
OUT = REPO / "THESIS" / "latex"
CHAPTERS = OUT / "chapters"

PLACEHOLDERS = {
    "[dataset_size]": "150.8 million",
    "[dataset_{size}]": r"150.8 \times 10^6",
    "[dataset_size_unique]": "2.0 million",
    "[test_fraction]": r"1\%",
    "[batch_size]": r"10{,}000",
    "[lr_start]": r"$10^{-2}$",
    "[lr_end]": r"$10^{-3}$",
    "[d_in]": "844",
    "[d_{in}]": "844",
    "[W]": "128",
    "[H]": "256",
    "[model_size]": r"174{,}723",
    "[keep_prob]": r"5\%",
    "[P_head]": r"66{,}563",
    "[P_{head}]": r"66{,}563",
    "[n_params]": r"174{,}723",
}

NOTE_TAGS = {"todo", "idea", "nota", "note", "fun", "important", "question", "review", "fix"}

# Raw unicode inside math mode (only these survive in $...$).
MATH_UNICODE = {
    "\u2212": "-",
    "\u2013": "-",
    "\u2014": "-",
    "\u03B1": r"\alpha", "\u03B2": r"\beta", "\u03B3": r"\gamma",
    "\u03B4": r"\delta", "\u03B5": r"\varepsilon", "\u03B6": r"\zeta",
    "\u03B7": r"\eta", "\u03B8": r"\theta", "\u03B9": r"\iota",
    "\u03BA": r"\kappa", "\u03BB": r"\lambda", "\u03BC": r"\mu",
    "\u03BD": r"\nu", "\u03BE": r"\xi", "\u03C0": r"\pi",
    "\u03C1": r"\rho", "\u03C3": r"\sigma", "\u03C4": r"\tau",
    "\u03C5": r"\upsilon", "\u03C6": r"\phi", "\u03C7": r"\chi",
    "\u03C8": r"\psi", "\u03C9": r"\omega",
    "\u0393": r"\Gamma", "\u0394": r"\Delta", "\u0398": r"\Theta",
    "\u039B": r"\Lambda", "\u039E": r"\Xi", "\u03A0": r"\Pi",
    "\u03A3": r"\Sigma", "\u03A6": r"\Phi", "\u03A8": r"\Psi",
    "\u03A9": r"\Omega",
    "\u2207": r"\nabla", "\u2202": r"\partial",
    "\u2208": r"\in", "\u2209": r"\notin", "\u2200": r"\forall",
    "\u2203": r"\exists", "\u2205": r"\emptyset",
    "\u2211": r"\sum", "\u220F": r"\prod", "\u222B": r"\int",
    "\u2264": r"\leq", "\u2265": r"\geq", "\u2260": r"\neq",
    "\u2248": r"\approx", "\u2261": r"\equiv", "\u221D": r"\propto",
    "\u2192": r"\to", "\u2190": r"\leftarrow", "\u2194": r"\leftrightarrow",
    "\u2191": r"\uparrow", "\u2193": r"\downarrow",
    "\u00D7": r"\times", "\u00B1": r"\pm", "\u00B7": r"\cdot",
    "\u221E": r"\infty", "\u2016": r"\Vert", "\u22C5": r"\cdot",
    "\u2026": r"\ldots", "\u221A": r"\sqrt",
    "\u27E8": r"\langle", "\u27E9": r"\rangle",
    "\u2329": r"\langle", "\u232A": r"\rangle",
    "\u2009": " ", "\u00A0": " ",
}

# Raw unicode in running text -> wrapped as inline math where needed.
TEXT_UNICODE = {
    "\u2212": "-",
    "\u2013": "--",
    "\u2014": "---",
    "\u2192": r"$\to$",
    "\u2190": r"$\leftarrow$",
    "\u2194": r"$\leftrightarrow$",
    "\u2248": r"$\approx$",
    "\u2264": r"$\leq$",
    "\u2265": r"$\geq$",
    "\u2260": r"$\neq$",
    "\u00D7": r"$\times$",
    "\u00B1": r"$\pm$",
    "\u221E": r"$\infty$",
    "\u2016": r"$\Vert$",
    "\u2026": r"\ldots{}",
    "\u2018": "`",
    "\u2019": "'",
    "\u201C": "``",
    "\u201D": "''",
    "\u2009": " ",
    "\u00A0": " ",
    "\u00B7": r"$\cdot$",
    "\u03B1": r"$\alpha$", "\u03B2": r"$\beta$", "\u03B3": r"$\gamma$",
    "\u03B4": r"$\delta$", "\u03B5": r"$\varepsilon$", "\u03B8": r"$\theta$",
    "\u03BB": r"$\lambda$", "\u03BC": r"$\mu$", "\u03C3": r"$\sigma$",
    "\u03C6": r"$\phi$",
}

# invisible characters that must vanish
INVISIBLE = {
    "\u200B": "", "\u2060": "", "\u2061": "", "\u200C": "", "\u200D": "", "\uFEFF": "",
}

# math display: $$ ... $$ -> \[ ... \]
DISPLAY_RE = re.compile(r"\$\$(.+?)\$\$", re.S)

# inline tokens to protect: display math, inline math, code spans
TOKEN_RE = re.compile(r"(\\\[.*?\\\]|\$[^$]+\$|`[^`]+`)", re.S)


def _apply_map(s: str, mapping: dict) -> str:
    for k, v in mapping.items():
        s = s.replace(k, v)
    return s


def _escape_plain(s: str) -> str:
    s = s.replace("&", r"\&")
    s = s.replace("%", r"\%")
    s = s.replace("#", r"\#")
    s = s.replace("_", r"\_")
    return s


def _transform_plain(s: str) -> str:
    # bold
    s = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", s)
    # italic with *...*
    s = re.sub(r"(?<!\*)\*(?!\*)([^*]+?)(?<!\*)\*(?!\*)", r"\\emph{\1}", s)
    # italic with _..._ (word-boundary underscores only)
    s = re.sub(r"(?<![\w\\])_([^_\s][^_]*?)_(?![\w])", r"\\emph{\1}", s)
    # text unicode -> latex
    s = _apply_map(s, TEXT_UNICODE)
    s = _escape_plain(s)
    return s


def inline(text: str) -> str:
    text = _apply_map(text, INVISIBLE)
    parts = TOKEN_RE.split(text)
    out = []
    for p in parts:
        if not p:
            continue
        if p.startswith("\\["):
            out.append(p)
        elif p.startswith("$"):
            out.append("$" + _apply_map(p[1:-1], MATH_UNICODE) + "$")
        elif p.startswith("`"):
            out.append(r"\texttt{" + _escape_plain(p[1:-1]) + "}")
        else:
            out.append(_transform_plain(p))
    return "".join(out)


def _caption_of(lines: list[str], i: int) -> tuple[str | None, int, str]:
    """If line i starts with a '*Table ... — caption*' marker, return
    (caption, i+1, trailing_prose). Trailing prose is any text after the
    closing '*' on the same line (e.g. Table 5.1)."""
    s = lines[i].strip()
    m = re.match(r"^\*Table\s+[\d.]+\s*[—-]\s*(.+?)\*", s)
    if m:
        cap = m.group(1).strip()
        rest = s[m.end():].strip()
        return cap, i + 1, rest
    return None, i, ""


def _width_score(header: list[str], data: list[list[str]]) -> int:
    ncol = len(header)
    score = 0
    for c in range(ncol):
        colmax = 0
        for r in [header] + data:
            if c < len(r):
                colmax = max(colmax, len(r[c]))
        score += colmax
    return score


def _table_to_latex(block: list[str], caption: str | None) -> list[str]:
    rows = []
    for raw in block:
        line = raw.strip()
        if not line.startswith("|"):
            continue
        cells = re.split(r"(?<!\\)\|", line.strip("|"))
        cells = [c.strip() for c in cells]
        rows.append(cells)
    if not rows:
        return []
    header = rows[0]
    sep = rows[1] if len(rows) > 1 else None
    data = rows[2:] if len(rows) > 2 else []

    # column alignment from the separator row
    if sep:
        aligns = []
        for c in sep:
            c = c.strip().strip(":").strip()
            if c.startswith(":") and c.endswith(":"):
                aligns.append("c")
            elif c.startswith(":"):
                aligns.append("l")
            elif c.endswith(":"):
                aligns.append("r")
            else:
                aligns.append("l")
    else:
        aligns = ["l"] * len(header)
    colspec = "".join(aligns)

    wide = len(header) >= 6 or _width_score(header, data) >= 100

    body = ["\\begin{tabular}{" + colspec + "}"]
    body.append("\\toprule")
    body.append(" & ".join(inline(c) for c in header) + r" \\")
    body.append("\\midrule")
    for r in data:
        body.append(" & ".join(inline(c) for c in r) + r" \\")
    body.append("\\bottomrule")
    body.append("\\end{tabular}")

    lines = []
    if caption:
        lines.append("\\begin{table}[H]")
        lines.append("\\centering")
        lines.append("\\caption{" + inline(caption) + "}")
        if wide:
            lines.append("\\resizebox{\\textwidth}{!}{")
            lines.extend("  " + b for b in body)
            lines.append("}")
        else:
            lines.append("\\small")
            lines.extend(body)
        lines.append("\\end{table}")
    else:
        lines.append("\\begin{center}")
        if wide:
            lines.append("\\resizebox{\\textwidth}{!}{")
            lines.extend("  " + b for b in body)
            lines.append("}")
        else:
            lines.append("\\small")
            lines.extend(body)
        lines.append("\\end{center}")
    return lines


def _title_strip(title: str) -> str:
    # strip manual "1.2.3 " numbering so LaTeX can auto-number
    return re.sub(r"^\d+(\.\d+)*\s+", "", title.strip())


def convert(text: str, *, is_abstract: bool = False) -> str:
    # 1. drop AI-note blockquotes, todo/idea/... tags, wiki links, strikethrough, rules
    lines = text.split("\n")
    kept = []
    for line in lines:
        s = line.rstrip()
        # blockquote notes
        if s.lstrip().startswith(">"):
            continue
        # horizontal rules
        if re.match(r"^\s*-{3,}\s*$", s) or re.match(r"^\s*\*{3,}\s*$", s):
            continue
        # note tags: #todo, #idea, ... (no space after #)
        m = re.match(r"^\s*#([a-z]+)\b", s)
        if m and m.group(1).lower() in NOTE_TAGS:
            continue
        # wiki links
        s = re.sub(r"\[\[[^\]]*\]\]", "", s).strip()
        kept.append(s)

    text = "\n".join(kept)

    # 2. remove gray span labels (keep following text)
    text = re.sub(r'<span style="color: #808080;">[^<]*</span>', "", text)
    # remove any other inline html tags (div/img handled later)
    text = re.sub(r"</?div[^>]*>", "", text)

    # 3. strikethrough -> drop
    text = re.sub(r"~~.*?~~", "", text, flags=re.S)

    # 3b. inline #tag notes (parenthetical and standalone)
    note_tag = r"(?:todo|idea|nota|note|fun|important|question|review|fix)"
    text = re.sub(r"\(\s*#" + note_tag + r"\b[^)]*\)", "", text)
    text = re.sub(r"(?<![\w])#" + note_tag + r"\b[^.!?\n]*[.!?]", "", text)

    # 4. placeholders
    for k, v in PLACEHOLDERS.items():
        text = text.replace(k, v)

    # 5. display math: $$...$$ -> \[...\], then protect all \[...\] blocks
    text = DISPLAY_RE.sub(lambda m: r"\[" + m.group(1) + r"\]", text)
    display_blocks: list[str] = []

    def _protect(m: re.Match) -> str:
        display_blocks.append(m.group(0))
        return f"@@DISPLAY{len(display_blocks) - 1}@@"

    text = re.sub(r"\\\[.*?\\\]", _protect, text, flags=re.S)

    # 6. structural conversion
    out_lines: list[str] = []
    raw_lines = text.split("\n")
    i = 0
    n = len(raw_lines)
    while i < n:
        s = raw_lines[i]
        st = s.strip()
        # image
        img = re.search(r'<img src="([^"]+)"[^>]*width="(\d+)"?[^>]*>', st)
        if img:
            src = img.group(1)
            w = int(img.group(2))
            scale = min(max(w / 1000.0, 0.3), 1.0)
            out_lines.append("\\begin{figure}[H]")
            out_lines.append("\\centering")
            out_lines.append(f"\\includegraphics[width={scale:.2f}\\textwidth]{{{src}}}")
            out_lines.append("\\end{figure}")
            out_lines.append("")
            i += 1
            continue
        if st == "":
            out_lines.append("")
            i += 1
            continue
        # table block
        if st.startswith("|"):
            block = []
            j = i
            while j < n and raw_lines[j].strip().startswith("|"):
                block.append(raw_lines[j])
                j += 1
            # skip blank lines, then look for a caption line
            jc = j
            while jc < n and raw_lines[jc].strip() == "":
                jc += 1
            caption, j2, trailing = _caption_of(raw_lines, jc)
            out_lines.extend(_table_to_latex(block, caption))
            out_lines.append("")
            if trailing:
                out_lines.append(inline(trailing))
                out_lines.append("")
            i = j2 if caption else j
            continue
        # headings
        if st.startswith("#### "):
            out_lines.append("\\subsubsection{" + inline(_title_strip(st[5:])) + "}")
            i += 1
            continue
        if st.startswith("### "):
            out_lines.append("\\subsection{" + inline(_title_strip(st[4:])) + "}")
            i += 1
            continue
        if st.startswith("## "):
            out_lines.append("\\section{" + inline(_title_strip(st[3:])) + "}")
            i += 1
            continue
        if st.startswith("# "):
            title = _title_strip(st[2:])
            if is_abstract:
                out_lines.append("\\chapter*{" + inline(title) + "}")
            else:
                out_lines.append("\\chapter{" + inline(title) + "}")
            i += 1
            continue
        # lists
        if re.match(r"^[-*] ", st):
            items = []
            while i < n and re.match(r"^[-*] ", raw_lines[i].strip()):
                items.append(re.sub(r"^[-*] ", "", raw_lines[i].strip()))
                i += 1
            out_lines.append("\\begin{itemize}")
            for it in items:
                out_lines.append("  \\item " + inline(it))
            out_lines.append("\\end{itemize}")
            out_lines.append("")
            continue
        if re.match(r"^\d+\. ", st):
            items = []
            while i < n and re.match(r"^\d+\. ", raw_lines[i].strip()):
                items.append(re.sub(r"^\d+\. ", "", raw_lines[i].strip()))
                i += 1
            out_lines.append("\\begin{enumerate}")
            for it in items:
                out_lines.append("  \\item " + inline(it))
            out_lines.append("\\end{enumerate}")
            out_lines.append("")
            continue
        # regular paragraph
        out_lines.append(inline(st))
        i += 1
    result = "\n".join(out_lines)
    result = re.sub(r"@@DISPLAY(\d+)@@", lambda m: display_blocks[int(m.group(1))], result)
    return result


CHAPTERS_ORDER = [
    ("1 - Introduction.md", "introduction", False),
    ("2 - Background.md", "background", False),
    ("3 - Method.md", "method", False),
    ("4 - Implementation.md", "implementation", False),
    ("5 - Results.md", "results", False),
    ("6 - Discussion.md", "discussion", True),
    ("7 - Conclusion.md", "conclusion", True),
    ("Appendix A.md", "appendix", False),
]

ABSTRACT_SRC = "0 - Abstract.md"


def build() -> None:
    CHAPTERS.mkdir(parents=True, exist_ok=True)

    # abstract
    abstract_md = (MD / ABSTRACT_SRC).read_text(encoding="utf-8")
    # strip the title line (first "# ...") and the trailing wiki link
    abstract_md = re.sub(r"^#\s.*$", "", abstract_md, count=1, flags=re.M)
    tex = convert(abstract_md, is_abstract=True)
    (CHAPTERS / "abstract.tex").write_text(tex, encoding="utf-8")

    for src, name, promote in CHAPTERS_ORDER:
        md = (MD / src).read_text(encoding="utf-8")
        if promote:
            # these files use `##` for the chapter title and `###` for sections
            md = re.sub(r"^###\s", "@@SUB@@", md, flags=re.M)
            md = re.sub(r"^##\s", "# ", md, flags=re.M)
            md = md.replace("@@SUB@@", "## ")
        tex = convert(md, is_abstract=False)
        (CHAPTERS / f"{name}.tex").write_text(tex, encoding="utf-8")
        print(f"converted {src} -> chapters/{name}.tex")

    main_tex = MAIN_TEMPLATE
    (OUT / "main.tex").write_text(main_tex, encoding="utf-8")
    print("wrote main.tex")


MAIN_TEMPLATE = r"""\documentclass[a4paper,11pt,oneside,openany]{book}

\usepackage{fontspec}
\setmainfont{DejaVu Serif}
\setmonofont{DejaVu Sans Mono}

\usepackage{amsmath,amssymb,amsfonts}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{array}
\usepackage{multirow}
\usepackage{float}
\usepackage{xcolor}
\usepackage{hyperref}
\usepackage{fancyhdr}
\usepackage{textcomp}
\usepackage{csquotes}

\graphicspath{{../../}}

\hypersetup{
    colorlinks=true,
    linkcolor=blue,
    citecolor=blue,
    urlcolor=blue,
    pdftitle={Unsupervised State Bucketing for Mixture of Experts in Resource-Constrained Chess Engines},
    pdfauthor={Omar Cusma Fait}
}

\setlength{\headheight}{30pt}
\pagestyle{fancy}
\fancyhf{}
\chead{\leftmark}
\rhead{\thepage}
\renewcommand{\headrulewidth}{0.4pt}

\title{Unsupervised State Bucketing for Mixture of Experts in Resource-Constrained Chess Engines}
\author{Omar Cusma Fait}

\begin{document}
\maketitle
\tableofcontents

\frontmatter
\input{chapters/abstract}

\mainmatter
\input{chapters/introduction}
\input{chapters/background}
\input{chapters/method}
\input{chapters/implementation}
\input{chapters/results}
\input{chapters/discussion}
\input{chapters/conclusion}

\appendix
\input{chapters/appendix}

\end{document}
"""


def compile_pdf() -> None:
    tectonic = "/tmp/opencode/tectonic"
    cmd = [tectonic, "-X", "compile", "main.tex"]
    print("compiling with tectonic ...")
    proc = subprocess.run(cmd, cwd=str(OUT), capture_output=True, text=True)
    print(proc.stdout[-4000:])
    print(proc.stderr[-4000:])
    if proc.returncode != 0:
        print(f"tectonic failed rc={proc.returncode}")
        return
    src = OUT / "main.pdf"
    dst = REPO / "main.pdf"
    if src.is_file():
        import shutil
        shutil.copy(src, dst)
        print(f"wrote {dst}")
    else:
        print("main.pdf not produced")


def main() -> None:
    build()
    compile_pdf()


if __name__ == "__main__":
    main()
