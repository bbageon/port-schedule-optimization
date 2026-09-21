"""Audit the integrated document only; no simulations, training, network or git."""
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import unquote

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
BUILD = ROOT / 'tmp/pdfs/yr317-integration'
RESPONSE = '23-review-response-draft.md'
REQUIRED_LABELS = {'sec:arch-models', 'sec:arch-commit', 'sec:user-context',
    'sec:env', 'sec:independent-protocol', 'sec:exp-paired', 'sec:exp-decomp',
    'sec:exp-learning', 'sec:pending-evidence', 'sec:concl'}
REQUIRED_PENDING = {'simulator deadlock repair and re-run', 'full-candidate ranking validation',
    'acceptance-network ablation', 'additional robustness analyses',
    'online latency measurements', 'simulator calibration against terminal measurements'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def uncomment(text):
    return re.sub(r'(?<!\\)%[^\n]*', '', text)


def bibliography(text):
    entries = re.findall(r'\\bibitem\{([^}]+)\}(.*?)(?=\\bibitem|\\end\{thebibliography\})',
                         uncomment(text), flags=re.S)
    return [(key, re.sub(r'\s+', '', body)) for key, body in entries]


def main_table(text):
    blocks = re.findall(r'\\begin\{table\}.*?\\end\{table\}', uncomment(text), re.S)
    matching = [b for b in blocks if r'\label{tab:main}' in b]
    if len(matching) != 1:
        raise ValueError('Exactly one historical performance table is required')
    return matching[0]


def policy_key(text):
    name = re.sub(r'\\[A-Za-z]+|[{}$]', '', text).strip().lower()
    if name in ('learned policy', 'full'):
        return 'full'
    if name in ('learned, time only', 'time-only'):
        return 'time-only'
    if name in ('learned, block only', 'block-only'):
        return 'block-only'
    return name.replace('first-come-first-served', 'first-come-first-serve')


def performance_rows(text):
    rows, column_counts = {}, []
    for raw in re.split(r'\\\\(?:\[[^]]*\])?', main_table(text)):
        cells = raw.split('&')
        if len(cells) < 4 or r'\%' not in cells[2]:
            continue
        # TeX rule commands preceding the row do not form part of its policy name.
        name = re.sub(r'\\(?:toprule|midrule|bottomrule)\s*', '', cells[0]).strip()
        key = policy_key(name)
        number = re.sub(r'\\[A-Za-z]+|[{}$\\%\s]', '', cells[2])
        cost = str(Decimal(number))
        wins = re.sub(r'\s+', '', cells[3])
        if not re.fullmatch(r'---|\d+/28', wins) or key in rows:
            raise ValueError('Duplicate policy or malformed historical daily wins')
        rows[key] = dict(actions=cells[1].strip(), reduction_percent=cost, daily_wins=wins)
        column_counts.append(len(cells))
    return rows, column_counts


def warning_groups(log):
    known, other, errors = [], [], []
    lines = log.splitlines()
    for index, line in enumerate(lines):
        context = '\n'.join(lines[index:index+4])
        if (re.search(r'^!|Overfull \\[hv]box|Underfull \\[hv]box', line)
                or re.search(r'undefined|(?:LaTeX|Package .*|pdfTeX) Error|Fatal error|Emergency stop', line, re.I)):
            errors.append(dict(line=index+1, text=context))
        if 'Warning:' in line:
            if (r'Package amsmath Warning: Unable to redefine math accent \vec.' in line
                    or 'Package caption Warning: Unknown document class (or package)' in line):
                known.append(dict(line=index+1, text=context))
            else:
                other.append(dict(line=index+1, text=context))
    return dict(known_class_warnings=known, other_warnings=other, errors=errors)


def audit():
    checks, details, exceptions = {}, {}, {}

    def check(name, action):
        try:
            value = action()
            checks[name] = bool(value)
        except Exception as exc:
            checks[name] = False
            exceptions[name] = f'{type(exc).__name__}: {exc}'

    required = ['main.tex', 'main.pdf', 'main-revised.tex', 'main-revised.pdf',
                'main-integrated.tex', 'main-integrated.pdf', 'integrated-revision.json', RESPONSE]
    details['missing_required_files'] = [name for name in required if not (OUT/name).is_file()]
    checks['required_document_files_exist'] = not details['missing_required_files']
    source = (OUT/'main.tex').read_text(encoding='utf-8') if (OUT/'main.tex').exists() else ''
    prior = (OUT/'main-revised.tex').read_text(encoding='utf-8') if (OUT/'main-revised.tex').exists() else ''
    draft = (OUT/'main-integrated.tex').read_text(encoding='utf-8') if (OUT/'main-integrated.tex').exists() else ''
    receipt = read(OUT/'integrated-revision.json') if (OUT/'integrated-revision.json').exists() else {}
    response = (OUT/RESPONSE).read_text(encoding='utf-8') if (OUT/RESPONSE).exists() else ''
    for filename, field in (('main.tex', 'source_sha256'), ('main-revised.tex', 'prior_draft_sha256'),
                            ('main-integrated.tex', 'draft_sha256')):
        check(filename+'_receipt_hash', lambda n=filename, f=field: sha(OUT/n) == receipt[f])
    check('original_pdf_preserved', lambda: sha(OUT/'main.pdf') == sha(OUT.parent/'submissionv2/main.pdf'))
    check('prior_pdf_preserved', lambda: sha(OUT/'main-revised.pdf') == read(OUT/'narrative-validation.json')['pdf_sha256'])
    check('original_and_prior_receipts_agree', lambda: receipt['source_sha256'] == read(
        OUT/'narrative-revision.json')['source_sha256'] and receipt['prior_draft_sha256'] == read(
        OUT/'narrative-revision.json')['draft_sha256'])
    check('explicit_pending_not_submission_ready', lambda: (receipt['submission_ready'] is False
          or (receipt.get('camera_ready') is True and 'Working draft' not in draft))
          and receipt['new_experiments'] == 0 and REQUIRED_PENDING <= set(receipt['unresolved']))
    for field, relative in (('prereg_sha256', 'outputs/reports/yr317_v3_independent_eval/prereg.md'),
                             ('addon_prereg_sha256', 'outputs/reports/yr317_v3_block_only/prereg.md')):
        check(field+'_unchanged', lambda f=field, p=relative: sha(ROOT/p) == receipt[f])

    draft_bib, prior_bib, original_bib = bibliography(draft), bibliography(prior), bibliography(source)
    keys = [key for key, _ in draft_bib]
    citations = [key.strip() for group in re.findall(r'\\cite\w*\{([^}]+)\}', uncomment(draft))
                 for key in group.split(',')]
    checks['twenty_two_unique_references'] = len(keys) == len(set(keys)) == receipt.get('references') == 22
    replacements = receipt.get('bibliography_replacements', {})
    prior_map, draft_map = dict(prior_bib), dict(draft_bib)
    details['bibliography_replacements'] = replacements
    checks['prior_bibliography_preserved'] = (bool(prior_bib)
        and set(draft_map) == {replacements.get(k, k) for k in prior_map}
        and all(draft_map[k] == v for k, v in prior_map.items() if k not in replacements))
    checks['original_references_retained'] = bool(original_bib) and set(dict(original_bib)) <= set(keys)
    checks['all_citations_defined_and_references_used'] = bool(citations) and set(citations) == set(keys)
    details['undefined_citations'] = sorted(set(citations)-set(keys))
    details['first_citation_order_matches_bibliography'] = list(dict.fromkeys(citations)) == keys
    labels = re.findall(r'\\label\{([^}]+)\}', uncomment(draft))
    refs = re.findall(r'\\(?:ref|eqref|autoref|pageref)\{([^}]+)\}', uncomment(draft))
    checks['unique_labels_and_required_sections'] = len(labels) == len(set(labels)) and REQUIRED_LABELS <= set(labels)
    checks['all_cross_references_defined'] = bool(refs) and set(refs) <= set(labels)
    details['undefined_cross_references'] = sorted(set(refs)-set(labels))
    details['duplicate_labels'] = [key for key, count in Counter(labels).items() if count > 1]

    def figures_preserved():
        extract = lambda text: re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}', uncomment(text))
        figures = extract(draft)
        details['figures'] = figures
        removed = receipt.get('figures_removed', [])
        details['figures_removed'] = removed
        expected = [f for f in extract(source) if f not in removed]
        if not figures or figures != expected or figures != [f for f in extract(prior) if f not in removed]:
            return False
        return all(sha(OUT/'figures'/name) == sha(OUT.parent/'submissionv2/figures'/name) for name in figures)
    check('original_figure_files_and_sequence_preserved', figures_preserved)

    def historical_table_preserved():
        original_rows, _ = performance_rows(source)
        rows, columns = performance_rows(draft)
        details['historical_performance_rows'] = rows
        return len(rows) == len(original_rows) == 11 and rows == original_rows and set(columns) == {4}
    check('eleven_policy_costs_and_daily_wins_preserved_without_p_columns', historical_table_preserved)
    # The eleven table rows above are fixed. Counts appearing solely in withdrawn
    # learning-significance arguments need not be reintroduced with those claims.
    checks['historical_time_rule_daily_win_example_retained'] = bool(source) and (
        '8/28' not in source or '8/28' in draft)
    details['daily_count_mentions_not_retained'] = sorted(set(re.findall(r'\b\d+/28\b', source))
                                                        - set(re.findall(r'\b\d+/28\b', draft)))
    pending = re.search(r'\\label\{sec:pending-evidence\}(.*?)(?=\\section\{|\Z)', uncomment(draft), re.S)
    pending_text = pending.group(1).lower() if pending else ''
    checks['pending_evidence_explicit_in_manuscript'] = bool(pending) and all(
        token in pending_text for token in ('deadlock', 'ranking', 'acceptance', 'latency', 'simulator', 'calibrat'))
    protocol = re.search(r'\\label\{sec:independent-protocol\}(.*?)(?=\\section\{|\Z)', uncomment(draft), re.S)
    protocol_text = re.sub(r'[{}\\\s]', '', protocol.group(1)).lower() if protocol else ''
    checks['registered_monthly_statistics_present'] = bool(protocol) and all(
        token in protocol_text for token in ('20,000', '9,900,721', '9,900,723', '97.5%', '95%', '80', 'paired', 'month'))
    independent = re.search(r'\\label\{sec:exp-independent\}(.*?)(?=\\subsection\{|\\section\{|\Z)', uncomment(draft), re.S)
    independent_text = independent.group(1) if independent else ''
    checks['independent_results_reported_with_deadlock_disclosure'] = bool(independent) and all(
        token in independent_text for token in ('16 months', '4 and Block-only in 3', 'deadlock', '15/15', 'tab:independent'))

    broken = []
    for target in re.findall(r'\]\(([^)]+)\)', response):
        if target.startswith(('https://', 'http://', '#')):
            continue
        local = unquote(target.split('#', 1)[0].strip('<>'))
        if local and not (OUT/local).exists():
            broken.append(target)
    details['response_markdown_lines'] = len(response.splitlines())
    details['broken_response_links'] = broken
    checks['response_links_and_line_limit'] = bool(response) and not broken and len(response.splitlines()) <= 200
    response_anchors = set(re.findall(r'`(sec:[^`]+)`', response))
    response_pages = re.findall(r'\u00a7(\d+(?:\.\d+)*)\s*\(p\.\s*(\d+)\)', response)
    def response_locations():
        if response_anchors and not response_anchors <= set(labels):
            return False
        if response_pages:
            aux = (BUILD/'main-integrated.aux').read_text(encoding='utf-8', errors='replace')
            defined = {(number, page) for label, number, page in re.findall(
                r'\\newlabel\{([^}]+)\}\{\{([^}]*)\}\{([^}]*)\}', aux) if label.startswith('sec:')}
            details['unmatched_response_pages'] = sorted(set(response_pages)-defined)
            return not details['unmatched_response_pages']
        return bool(response_anchors)
    check('response_section_locations_exist', response_locations)
    checks['response_covers_all_reviewers'] = all(x in response for x in (
        'comment 1', 'comment 2', 'comment 3', 'comment 4a', 'comment 4b',
        'comment 5a', 'comment 5b', 'Reviewer 2', 'Reviewer 3'))

    def compile_check():
        log_path = BUILD/'main-integrated.log'
        log = log_path.read_text(encoding='utf-8', errors='replace')
        groups = warning_groups(log)
        details['compile_log'] = str(log_path.relative_to(ROOT)).replace('\\', '/')
        details.update(groups)
        return not groups['errors'] and not groups['other_warnings'] and 'Output written on' in log
    check('compile_finished_without_layout_reference_or_unexpected_warnings', compile_check)
    check('official_pdf_matches_final_build', lambda: sha(OUT/'main-integrated.pdf') == sha(BUILD/'main-integrated.pdf'))
    check('build_is_not_older_than_source', lambda: (BUILD/'main-integrated.pdf').stat().st_mtime >=
          (OUT/'main-integrated.tex').stat().st_mtime)

    def inspect_pdf():
        if not (OUT/'main-integrated.pdf').is_file():
            raise FileNotFoundError('Final main-integrated.pdf is missing; the author must copy the reviewed build explicitly')
        import pymupdf
        outside, page_texts = [], []
        with pymupdf.open(OUT/'main-integrated.pdf') as pdf:
            details['pdf_pages'] = len(pdf)
            for number, page in enumerate(pdf, start=1):
                text = page.get_text()
                page_texts.append(text)
                for word in page.get_text('words'):
                    if not page.rect.contains(pymupdf.Rect(word[:4])):
                        outside.append(dict(page=number, word=word[4]))
        details['text_outside_page'] = outside
        details['pdf_sha256'] = sha(OUT/'main-integrated.pdf')
        details['target_page_limit_met'] = details['pdf_pages'] <= 12
        details['page_limit_status'] = 'within_target' if details['target_page_limit_met'] else 'over_target_requires_decision'
        return bool(page_texts) and all(text.strip() for text in page_texts) and not outside
    check('final_pdf_readable_without_text_outside_page', inspect_pdf)
    return dict(schema='yr317.integrated-document-validation.v1', at=datetime.now(timezone.utc).isoformat(),
        checks=checks, passed=all(checks.values()), exceptions=exceptions, details=details,
        target_pages=12, target_page_limit_is_hard_failure=False,
        scientific_validation='Document consistency only; pending experiments and historical costs are not newly validated performance.',
        new_experiments=0, submission_ready=False)


def main():
    result = audit()
    (OUT/'integrated-validation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(passed=result['passed'], failed_checks=[k for k,v in result['checks'].items() if not v],
        exceptions=result['exceptions'], pdf_pages=result['details'].get('pdf_pages'),
        page_limit_status=result['details'].get('page_limit_status', 'missing_pdf'), submission_ready=False), ensure_ascii=False))
    return int(not result['passed'])


if __name__ == '__main__':
    raise SystemExit(main())
