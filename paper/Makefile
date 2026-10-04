EMACS = emacs
LATEXMK ?= latexmk
PYTHON ?= python3

.PHONY: all check clean
all: check

build/paper.tex: paper.org export.el
	$(EMACS) --batch -Q -l ./export.el paper.org --eval '(org-latex-export-to-latex)'

build/references.bib: references.bib
	mkdir -p build
	cp references.bib build/references.bib

build/paper.pdf: build/paper.tex build/references.bib
	$(LATEXMK) -cd -pdf -interaction=nonstopmode -halt-on-error build/paper.tex

check: build/paper.pdf
	$(PYTHON) check_layout.py build/paper.pdf

clean:
	$(LATEXMK) -cd -c build/paper.tex
