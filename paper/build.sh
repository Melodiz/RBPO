#!/usr/bin/env bash
# Build the preprint with pdfLaTeX (de-branded ICML 2025 style).
# Usage: ./build.sh   (run from the paper/ directory)
set -e
pdflatex -interaction=nonstopmode -halt-on-error paper.tex
bibtex paper
pdflatex -interaction=nonstopmode -halt-on-error paper.tex
pdflatex -interaction=nonstopmode -halt-on-error paper.tex
echo "Built paper.pdf"
