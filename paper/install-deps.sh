#!/usr/bin/env bash
# Install the tools needed by `make -C paper` (not the ML experiments).
set -euo pipefail

DRY_RUN=false

usage() {
    cat <<'HELP'
Usage: bash paper/install-deps.sh [--dry-run | --check]

Debian/Ubuntu (including WSL): apt packages; sudo when needed.
macOS: Homebrew tools plus MacTeX; installs Homebrew if missing.
Windows Git Bash/Cygwin: installs Ubuntu via WSL or runs this script in Ubuntu.
Other systems: reports an unsupported platform without changing it.

--dry-run  Show installation commands without executing them.
--check    Check installed build tools without installing anything.
HELP
}

run() {
    printf '+'
    printf ' %q' "$@"
    printf '\n'
    if [[ "$DRY_RUN" == false ]]; then
        "$@"
    fi
}

check_dependencies() {
    local tool missing=0
    for tool in emacs make python3 latexmk pdflatex bibtex kpsewhich pdfinfo pdftotext; do
        if command -v "$tool" >/dev/null 2>&1; then
            printf 'Found %s\n' "$tool"
        else
            printf 'Missing %s\n' "$tool" >&2
            missing=1
        fi
    done
    if command -v emacs >/dev/null 2>&1; then
        if ! emacs --batch -Q --eval \
            '(progn (require (quote org)) (when (version< (org-version) "9.5") (error "Org 9.5+ required")))'; then
            missing=1
        fi
    fi
    if command -v kpsewhich >/dev/null 2>&1; then
        for tool in IEEEtran.cls IEEEtran.bst fontenc.sty inputenc.sty cite.sty \
                    amsmath.sty amssymb.sty graphicx.sty booktabs.sty hyperref.sty ptmr8r.tfm; do
            if ! kpsewhich "$tool" >/dev/null; then
                printf 'Missing TeX component: %s\n' "$tool" >&2
                missing=1
            fi
        done
    fi
    return "$missing"
}

install_debian() {
    local -a elevate=()
    if [[ "$EUID" -ne 0 ]]; then
        if ! command -v sudo >/dev/null 2>&1 && [[ "$DRY_RUN" == false ]]; then
            printf 'sudo is required; alternatively run this script as root.\n' >&2
            return 1
        fi
        elevate=(sudo)
    fi
    run "${elevate[@]}" apt-get update
    run "${elevate[@]}" apt-get install -y \
        emacs make python3 latexmk poppler-utils \
        texlive-latex-base texlive-latex-recommended texlive-latex-extra \
        texlive-fonts-recommended texlive-publishers
}

install_macos() {
    local brew_cmd installer
    if [[ "$EUID" -eq 0 ]]; then
        printf 'Run this script as your normal macOS user, without sudo.\n' >&2
        return 1
    fi
    if command -v brew >/dev/null 2>&1; then
        brew_cmd=$(command -v brew)
    elif [[ -x /opt/homebrew/bin/brew ]]; then
        brew_cmd=/opt/homebrew/bin/brew
    elif [[ -x /usr/local/bin/brew ]]; then
        brew_cmd=/usr/local/bin/brew
    else
        printf 'Homebrew is missing; installing it using its official installer.\n'
        if [[ "$DRY_RUN" == true ]]; then
            printf '+ Download https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh and run with /bin/bash\n'
            brew_cmd=brew
        else
            installer=$(mktemp "${TMPDIR:-/tmp}/paper-homebrew.XXXXXX")
            if ! curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh -o "$installer"; then
                rm -f "$installer"
                return 1
            fi
            if ! /bin/bash "$installer"; then
                rm -f "$installer"
                return 1
            fi
            rm -f "$installer"
            if [[ -x /opt/homebrew/bin/brew ]]; then
                brew_cmd=/opt/homebrew/bin/brew
            else
                brew_cmd=/usr/local/bin/brew
            fi
        fi
    fi
    run "$brew_cmd" install emacs make python poppler
    # Full MacTeX includes IEEEtran, BibTeX, latexmk, and the required fonts/packages.
    run "$brew_cmd" install --cask mactex-no-gui
    if [[ "$DRY_RUN" == false ]]; then
        PATH="$("$brew_cmd" --prefix)/bin:$("$brew_cmd" --prefix make)/libexec/gnubin:/Library/TeX/texbin:$PATH"
        export PATH
        printf 'For future shells, enable Homebrew in PATH and include /Library/TeX/texbin.\n'
    fi
}

install_windows() {
    local script_dir windows_script linux_script
    if ! command -v wsl.exe >/dev/null 2>&1; then
        printf 'WSL is unavailable. In Administrator PowerShell run: wsl --install -d Ubuntu\nThen restart if requested, open Ubuntu, and run this script there.\n' >&2
        return 1
    fi
    if [[ "$DRY_RUN" == true ]]; then
        printf '+ If Ubuntu is not ready: wsl.exe --install -d Ubuntu --no-launch\n'
        printf '+ Once Ubuntu is ready: rerun this installer inside Ubuntu using wsl.exe\n'
        return 0
    fi
    if ! wsl.exe -d Ubuntu --exec true >/dev/null 2>&1; then
        if ! wsl.exe --install -d Ubuntu --no-launch; then
            printf 'Run wsl --install -d Ubuntu in Administrator PowerShell, then retry.\n' >&2
            return 1
        fi
        printf 'WSL setup started. Restart if requested, open Ubuntu and create your Linux user,\nthen rerun this script. Dependencies are not installed until that step completes.\n'
        return 2
    fi
    if ! command -v cygpath >/dev/null 2>&1; then
        printf 'Open Ubuntu and run this script there; Windows path conversion needs Git Bash or Cygwin.\n' >&2
        return 1
    fi
    script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
    windows_script=$(cygpath -w "$script_dir/install-deps.sh")
    linux_script=$(MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu --exec wslpath -u "$windows_script")
    linux_script=${linux_script//$'\r'/}
    MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu --exec bash "$linux_script"
}

main() {
    if [[ $# -gt 1 ]]; then usage >&2; return 1; fi
    case "${1:-}" in
        --dry-run) DRY_RUN=true ;;
        --check) check_dependencies; return ;;
        --help|-h) usage; return ;;
        '') ;;
        *) usage >&2; return 1 ;;
    esac
    case "$(uname -s)" in
        Linux)
            if [[ ! -r /etc/os-release ]]; then
                printf 'Cannot identify this Linux distribution.\n' >&2
                return 1
            fi
            # shellcheck disable=SC1091
            . /etc/os-release
            case " ${ID:-} ${ID_LIKE:-} " in
                *' debian '*|*' ubuntu '*) install_debian ;;
                *) printf 'Unsupported distribution: %s. See paper/README.md for required tools.\n' "${PRETTY_NAME:-Linux}" >&2; return 1 ;;
            esac
            ;;
        Darwin) install_macos ;;
        MINGW*|MSYS*|CYGWIN*) install_windows; return ;;
        *) printf 'Unsupported operating system. See paper/README.md for required tools.\n' >&2; return 1 ;;
    esac
    if [[ "$DRY_RUN" == false ]]; then
        check_dependencies
        printf '\nDependencies ready. From the repository root, run: make -C paper\n'
    fi
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
