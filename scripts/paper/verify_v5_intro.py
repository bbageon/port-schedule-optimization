"""Check that the editable introduction deck contains its declared content."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET
import zipfile


ROOT = Path(__file__).resolve().parents[2]
CONTENT = ROOT / 'docs/paper/v5/introduction/slides.json'
PPTX = ROOT / 'docs/paper/v5/introduction/v5_졸업논문_서론.pptx'
BUILD = ROOT / 'outputs/presentation-build/yr315-v5-intro-r4'
REPORT = ROOT / 'outputs/reports/yr315_v5_introduction/verification.json'
NS = {'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
      'p': 'http://schemas.openxmlformats.org/presentationml/2006/main'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(value):
    return re.sub(r'\s+', '', value)


def main():
    deck = json.loads(CONTENT.read_text(encoding='utf-8'))
    geometry = json.loads((BUILD / 'powerpoint-check.json').read_text(encoding='utf-8-sig'))
    assert not geometry['text_fit_issues']
    assert geometry['content_sha256'] == digest(CONTENT)
    assert len(deck['slides']) == 19 and len(deck['references']) == 7
    tables, notes, links = [], [], set()
    with zipfile.ZipFile(PPTX) as package, zipfile.ZipFile(BUILD / 'candidate.pptx') as draft:
        ppt_parts = [name for name in package.namelist() if name.startswith('ppt/')]
        assert all(package.read(name) == draft.read(name) for name in ppt_parts), 'Final slide data differs from inspected draft'
        for index, item in enumerate(deck['slides'], 1):
            slide = ET.fromstring(package.read(f'ppt/slides/slide{index}.xml'))
            actual = normalize(''.join(slide.itertext()))
            expected = [item['title'], *item.get('bullets', []), *item.get('paragraphs', [])]
            expected += [item[key] for key in ('takeaway', 'subtitle') if key in item]
            if 'table' in item:
                expected += item['table']['headers']
                expected += [cell for row in item['table']['rows'] for cell in row]
                assert len(slide.findall('.//a:tbl', NS)) == 1
                tables.append(index)
            for text in expected:
                assert normalize(text) in actual, (index, text)
            note_xml = ET.fromstring(package.read(f'ppt/notesSlides/notesSlide{index}.xml'))
            note_text = normalize(''.join(note_xml.itertext()))
            assert normalize(item['notes']) in note_text
            notes.append(index)
            relations = f'ppt/slides/_rels/slide{index}.xml.rels'
            for relation in ET.fromstring(package.read(relations)):
                if relation.attrib['Type'].endswith('/hyperlink'):
                    links.add(relation.attrib['Target'])
            assert (BUILD / f'slide-{index:02d}.png').is_file()
            assert digest(BUILD / f'final-slide-{index:02d}.png') == digest(BUILD / f'slide-{index:02d}.png')
        assert tables == [3, 5, 8, 10, 13]
        assert {ref['url'] for ref in deck['references']} <= links
    unchanged = subprocess.check_output(
        ['git', 'diff', '0df2aa4', '--', 'src/yard_rl/v5', 'docs/paper/v3'],
        cwd=ROOT, text=True, encoding='utf-8')
    assert not unchanged, 'Protected v5 code or v3 paper changed'
    report = {
        'task': 'YR-315', 'scope': 'introduction_coursework_only', 'status': 'passed',
        'slides': 19, 'introduction_slides_including_cover': 17, 'reference_slides': 2,
        'references': 7, 'editable_table_slides': tables, 'notes_slides': notes,
        'all_declared_text_present': True, 'reference_links_present': True,
        'text_fit_issues': 0, 'powerpoint_roundtrip': True,
        'final_presentation_parts_identical_to_rendered_draft': True,
        'final_pptx_reopened_and_rendered_slides': 19,
        'final_renders_identical_to_inspected_previews': True,
        'production_code_and_v3_paper_unchanged': True,
        'research_performance_claim': False,
        'authoring': 'Microsoft PowerPoint 16.0 native editable text and tables',
        'skill_runtime_fallback': 'load_workspace_dependencies and @oai/artifact-tool unavailable; no substitute installed',
        'artifact_sha256': {
            str(path.relative_to(ROOT)).replace('\\', '/'): digest(path)
            for path in [PPTX, CONTENT, PPTX.with_name('서론_문장형_초안.md')]
        },
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
