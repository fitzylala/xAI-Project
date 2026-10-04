# Paper

Edit `paper.org`; keep bibliography entries in `references.bib`.

## Install dependencies

From the repository root, run:

```sh
bash paper/install-deps.sh
```

The installer detects the operating system:

- **Debian/Ubuntu, including WSL:** installs the required apt packages, using
  sudo unless already running as root.
- **macOS:** installs Homebrew if missing, then Emacs, Make, Python, Poppler,
  and MacTeX without GUI applications. MacTeX is a large download that includes
  IEEEtran, latexmk, BibTeX, and the required fonts. Run without sudo; follow
  Homebrew's shell setup instructions and ensure `/Library/TeX/texbin` is on PATH.
- **Windows (Git Bash or Cygwin):** uses Ubuntu under WSL. If Ubuntu is ready,
  the script installs the packages there. Otherwise it starts WSL/Ubuntu setup;
  restart if requested, open Ubuntu to create a Linux user, and rerun the script.
  If administrator access is required, run `wsl --install -d Ubuntu` in
  Administrator PowerShell. Run subsequent paper builds inside Ubuntu.
- **Other systems:** stops with an unsupported-platform message.

Preview commands without installing anything, or verify an existing setup:

```sh
bash paper/install-deps.sh --dry-run
bash paper/install-deps.sh --check
```

The installer only installs paper-building tools; no Python packages or ML
dependencies are needed. It checks the commands, Org version, and TeX components
after installation. WSL initialization returns status 2 when a restart/user setup
is still needed; installation failures return a nonzero status.

Platform installation references: [Homebrew](https://brew.sh/),
[MacTeX cask](https://formulae.brew.sh/cask/mactex-no-gui), and
[Microsoft WSL instructions](https://learn.microsoft.com/en-us/windows/wsl/install).

## Build and check

From the repository root:

```sh
make -C paper
```

Requires Emacs with Org 9.5+, LaTeX with IEEEtran and its standard packages,
BibTeX, latexmk, Python 3, and Poppler (`pdfinfo` and `pdftotext`).
All generated files go into `paper/build/`: `paper.tex`, `paper.pdf`, a copy of
`references.bib`, and the LaTeX auxiliary files. Edit the original bibliography,
not the generated copy. The build checks that the PDF has at
most four pages, with references on one separate final page. The main text
therefore has at most three pages. The check does not assess writing quality,
citation relevance, or whether the draft sections have been completed.

## Emacs export

Load this directory's `export.el` once with `M-x load-file`, then open
`paper.org`. Export with `C-c C-e l l` for LaTeX or `C-c C-e l p` for PDF.
Loading `export.el` while the paper buffer is already open also works.
Both export commands write to `build/` and create that directory if needed.
Run `make -C paper check` from the repository root after exporting to check
the page budget; Emacs's standard export alone does not run that check.

## Format and scope

The exporter uses `IEEEtran` with `conference,10pt`, its default two-column
layout and Times-family font, and its unmodified margins. This is the LaTeX
IEEE conference template's Times Roman implementation, rather than an embedded
Microsoft Times New Roman font. No geometry or font-size overrides are used.

- [IEEE template page](https://www.ieee.org/conferences/publishing/templates.html)
- [IEEE conference template on Overleaf](https://www.overleaf.com/latex/templates/ieee-conference-template/grfzhhncsfqn)

For Overleaf, upload `build/paper.tex` and `build/references.bib` to the IEEE
conference template and use pdfLaTeX. Keep the local Org file as the source of
truth. The local page check can also be run on a downloaded PDF if its matching
`.aux` file is supplied alongside it.

The outline contains Abstract, Introduction, Background and Related Work,
Methodology, Results and Discussion, Conclusion and Limitations, and References.
Org comments give writing prompts without appearing in the PDF. The topic
remains the effect of Gini versus entropy on decision-tree interpretability
across the three selected medical datasets.

The `#+AUTHOR:` line lists the four group members in the README's team order.
For IEEE affiliation blocks, use
`#+LATEX_HEADER: \author{\IEEEauthorblockN{Names}\IEEEauthorblockA{Affiliation}}`
and change `author:t` to `author:nil`, so Org does not overwrite that block.

## Citations

Use Org citations, for example `[cite:@raileanu2004gini]` or
`[cite:@raileanu2004gini; @wang2016tsallis]`. Export uses BibTeX with the
`IEEEtran` bibliography style for numbered references in citation order.
Do not configure this paper to use BibLaTeX/Biber instead.

The draft includes `\nocite{*}` to show all ten collected references while
planning the page budget. Remove that line once the prose cites its sources;
then only cited entries will appear. Add relevant course-assigned readings to
`references.bib` and cite them alongside the group's additional sources. The
assigned reading list has not yet been supplied.

A `\clearpage` before the bibliography starts references on a new physical
page, flushing both columns and queued figures. Keep it in place. Do not reduce
margins or shrink text to meet the length limit; edit the content instead.
