#!/bin/sh
# Render WHITEPAPER.md to WHITEPAPER.pdf.
#
#   ./whitepaper/build.sh
#
# The PDF is a build artifact of the Markdown, and the Markdown is the source
# of truth. It exists as a script rather than a remembered command line because
# the two drifted once: the PDF is committed, so nothing fails when it goes
# stale, and a stale PDF is worse than none — it is a document someone may send
# to a reader, silently describing an older design.
#
# Run this in the same change set as any WHITEPAPER.md edit.
#
# Requires: pandoc and typst (brew install pandoc typst).

set -eu

cd "$(dirname "$0")"
SRC="WHITEPAPER.md"
OUT="WHITEPAPER.pdf"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT INT TERM

command -v pandoc >/dev/null 2>&1 || { echo "error: pandoc not found (brew install pandoc)" >&2; exit 1; }
command -v typst  >/dev/null 2>&1 || { echo "error: typst not found (brew install typst)"  >&2; exit 1; }

# -f gfm, not the default markdown: pandoc's default reader strips the leading
# numbers from heading ids ("1. Introduction" -> #introduction), which breaks
# every "#1-introduction" anchor in the table of contents.
pandoc -f gfm -t typst "$SRC" -o "$TMP/body.typ"

{
    cat <<'PREAMBLE'
#set document(title: "Loom: A Distributed Expert-Cache Architecture for Frontier Mixture-of-Experts Models", author: "Parthasarathy Ramanujam")
#set page(paper: "a4", margin: (x: 2.2cm, y: 2.4cm), numbering: "1")
#set text(font: ("Times New Roman", "Charter", "New York"), size: 10.5pt)
#set par(justify: true, leading: 0.62em)
#show heading: set block(above: 1.4em, below: 0.8em)
#show link: set text(fill: rgb("#1a4d8f"))
#show raw: set text(font: ("Menlo", "DejaVu Sans Mono", "Courier New"))
// Inline code sits in running text, so it tracks the body size. Block code
// does not: the widest figure in this document is 102 columns, and at the
// inline size that runs off the right margin and wraps into the gutter. Size
// the block variant to fit the measure instead.
#show raw.where(block: true): set text(size: 7.1pt)
#show raw.where(block: false): set text(size: 9pt)
#show raw.where(block: true): set block(breakable: true, width: 100%)

// pandoc's typst writer emits #horizontalrule for a `---` rule but leaves it
// to the template to define. Same definition pandoc's own default template
// uses, so the output matches what `pandoc -o out.pdf` would produce.
#let horizontalrule = line(start: (25%, 0%), end: (75%, 0%))

// Tables here are wide and often taller than a page. Typst blocks are
// unbreakable by default, so a long one overflows off the bottom instead of
// continuing overleaf -- that is what corrupted page 15 before.
#show figure: set block(breakable: true)
#show table: set block(breakable: true)
#set table(stroke: 0.4pt + luma(180), inset: 5pt)

PREAMBLE
    cat "$TMP/body.typ"
} > "$TMP/doc.typ"

typst compile --root "$TMP" "$TMP/doc.typ" "$OUT"

echo "wrote $(pwd)/$OUT ($(wc -c < "$OUT" | tr -d ' ') bytes)"
