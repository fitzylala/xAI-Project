"""Check the four-page budget and a separate final references page after export."""

from pathlib import Path
import re
import subprocess
import sys


def check_layout(pdf: Path) -> None:
    info = subprocess.check_output(["pdfinfo", str(pdf)], text=True)
    pages = int(re.search(r"^Pages:\s+(\d+)", info, re.MULTILINE).group(1))
    aux = pdf.with_suffix(".aux").read_text()
    match = re.search(r"\\newlabel\{paper:references\}\{\{[^}]*\}\{(\d+)\}", aux)
    if not match:
        raise SystemExit("Missing references-page label; rebuild from paper.org.")
    reference_page = int(match.group(1))
    if pages > 4:
        raise SystemExit(f"Page limit exceeded: {pages} pages (maximum 4, including references).")
    if reference_page < 2 or reference_page != pages:
        raise SystemExit("References must occupy one separate final page after the main text.")
    text = subprocess.check_output(
        ["pdftotext", "-f", str(reference_page), "-l", str(reference_page), str(pdf), "-"],
        text=True,
    )
    # IEEE small caps can extract as "R EFERENCES" in Poppler.
    first_line = text.lstrip().splitlines()[0] if text.strip() else ""
    if "".join(first_line.split()).upper() != "REFERENCES":
        raise SystemExit("The final page must begin with the References heading.")
    if not re.search(r"\[1\]", text):
        raise SystemExit("The references page contains no numbered bibliography entries.")
    print(f"Layout check passed: {pages} pages; references on separate final page {reference_page}.")


if __name__ == "__main__":
    check_layout(Path(sys.argv[1] if len(sys.argv) > 1 else "build/paper.pdf"))
