#!/bin/bash

# Check if argument is provided
if [ $# -eq 0 ]; then
	echo "Usage: $0 path/to/latex-report.tex"
	exit 1
fi

LATEXFILE="$1"

# Check if the LaTeX file exists
if [ ! -f "$LATEXFILE" ]; then
	echo "Error: File '$LATEXFILE' not found"
	exit 1
fi

# Generate PDF file: clean auxiliary files + compile LaTeX file

# Clean previous build files
latexmk -c "$LATEXFILE"

# Compile LaTeX file to PDF and watch for changes
# -pvc is removed because it opens Adobe Reader
# Find out why the code below does not work completely -> Something is wrong with the citations
latexmk -pdf -bibtex -f -interaction=nonstopmode -silent "$LATEXFILE"

# Clean up generated files manually cause I cannot be bothered to learn the LaTeX way
rm *.aux
rm *.bbl
rm *.blg
rm *.fdb_latexmk
rm *.fls
rm *.log

echo "PDF of LaTeX file is generated ..."