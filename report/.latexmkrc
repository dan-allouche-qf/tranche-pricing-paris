# latexmk configuration for the tranche-pricing-paris working paper.
#
# We use natbib + bibtex (not biblatex / biber). latexmk's default
# heuristic occasionally fails to detect the bibtex dependency from a
# clean state; this file forces the correct toolchain.
$pdf_mode = 1;
$bibtex_use = 2;
# Treat the first pdflatex pass as best-effort: it will report unresolved
# citations *before* bibtex has run, which is normal. We rely on
# latexmk's dependency loop to iterate until the build stabilises.
$pdflatex = "pdflatex -interaction=nonstopmode %O %S";
$force_mode = 1;
$max_repeat = 5;
