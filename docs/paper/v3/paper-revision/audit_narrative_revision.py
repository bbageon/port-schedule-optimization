"""Verify the limited narrative revision without treating old statistics as validated."""
import hashlib
import json
from pathlib import Path
import re

import pymupdf

OUT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    original = (OUT / "main.tex").read_text(encoding="utf-8")
    draft = (OUT / "main-revised.tex").read_text(encoding="utf-8")
    receipt = json.loads((OUT / "narrative-revision.json").read_text(encoding="utf-8"))
    section = lambda text: text.split("\\section{Results}", 1)[1].split("\\section{Conclusion}", 1)[0]
    tables = lambda text: re.findall(r"\\begin\{table\}.*?\\end\{table\}", text, re.S)
    figures = lambda text: re.findall(r"\\begin\{figure\}.*?\\end\{figure\}", text, re.S)
    cite_groups = [group.split(",") for group in re.findall(r"\\cite\{([^}]+)\}", draft)]
    cited = list(dict.fromkeys(key for group in cite_groups for key in group))
    bibliography = re.findall(r"\\bibitem\{([^}]+)\}", draft)
    positions = {key: i for i, key in enumerate(bibliography)}
    log = (OUT / "main-revised.log").read_text(encoding="utf-8", errors="replace")
    errors = re.findall(r"^!.*|^Overfull.*|^Underfull.*|^LaTeX Warning:.*undefined.*", log, re.M)
    with pymupdf.open(OUT / "main-revised.pdf") as pdf:
        pages = len(pdf)
        pdf_text = "\n".join(page.get_text() for page in pdf)
        chars_outside_page = []
        for i, page in enumerate(pdf):
            for word in page.get_text("words"):
                if not page.rect.contains(pymupdf.Rect(word[:4])):
                    chars_outside_page.append({"page": i + 1, "word": word[4]})
    checks = {
        "original_source_matches_baseline": (OUT / "main.tex").read_bytes() == (OUT.parent / "submissionv2/main.tex").read_bytes(),
        "original_pdf_matches_baseline": (OUT / "main.pdf").read_bytes() == (OUT.parent / "submissionv2/main.pdf").read_bytes(),
        "source_hash_matches_receipt": sha(OUT / "main.tex") == receipt["source_sha256"],
        "draft_hash_matches_receipt": sha(OUT / "main-revised.tex") == receipt["draft_sha256"],
        "results_section_unchanged": section(original) == section(draft),
        "all_original_tables_preserved": tables(original) == tables(draft),
        "all_original_figures_preserved": figures(original) == figures(draft),
        "citation_order": cited == bibliography and all(group == sorted(group, key=positions.get) for group in cite_groups),
        "reference_count": len(bibliography) == 22,
        "no_layout_or_reference_warnings": not errors,
        "no_pdf_text_outside_page": not chars_outside_page,
        "users_section_present": "Intended Users and Operational Context" in pdf_text,
        "partial_revision_explicit": receipt["submission_ready"] is False and bool(receipt["unresolved"]),
    }
    payload = {"schema": "yr317.narrative-revision-validation.v1", "checks": checks,
        "passed": all(checks.values()), "pdf_pages": pages, "target_pages": 12,
        "target_page_limit_met": pages <= 12, "warnings": errors,
        "pdf_sha256": sha(OUT / "main-revised.pdf"),
        "pdf_render_review": "all 14 pages in contact sheet; pages 6, 7, 12, 13 inspected separately",
        "scientific_validation": "unchanged results are preserved historical text, not newly validated statistics",
        "new_experiments": 0, "submission_ready": False}
    (OUT / "narrative-validation.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": payload["passed"], "checks": checks, "pdf_pages": pages,
                      "target_page_limit_met": pages <= 12}, ensure_ascii=False))
    return int(not payload["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
