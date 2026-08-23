"""
scripts/generate_pptx_from_md.py

Generate a PowerPoint (.pptx) from a simple markdown slide deck file (docs/pipeline_slides.md).

Usage:
  python scripts/generate_pptx_from_md.py docs/pipeline_slides.md docs/pipeline_slides.pptx

Requires: python-pptx
  pip install python-pptx

The script treats '---' as slide separators and plain text lines as bullet lines. Simple parsing suitable for the repository slide deck.
"""

import sys
from pptx import Presentation
from pptx.util import Inches, Pt


def md_to_slides(md_path):
    with open(md_path, 'r', encoding='utf-8') as f:
        lines = [l.rstrip('\n') for l in f]

    slides = []
    current = []
    for line in lines:
        if line.strip() == '---':
            if current:
                slides.append(current)
                current = []
        else:
            current.append(line)
    if current:
        slides.append(current)
    return slides


def add_slide(prs, title, bullets):
    slide_layout = prs.slide_layouts[1]  # Title and Content
    slide = prs.slides.add_slide(slide_layout)
    title_placeholder = slide.shapes.title
    body = slide.shapes.placeholders[1]

    title_placeholder.text = title
    tf = body.text_frame
    tf.clear()
    for i, b in enumerate(bullets):
        if i == 0:
            p = tf.paragraphs[0]
            p.text = b
        else:
            p = tf.add_paragraph()
            p.text = b
        p.level = 0
        p.font.size = Pt(18)


def make_pptx(md_path, pptx_path):
    slides = md_to_slides(md_path)
    prs = Presentation()

    for s in slides:
        # Determine title: first non-empty line
        title = ''
        bullets = []
        for line in s:
            line = line.strip()
            if not line:
                continue
            # Remove leading markdown list markers
            if line.startswith('#'):
                # Treat as slide title if it is like '# Title' or '## Title'
                cleaned = line.lstrip('#').strip()
                if not title:
                    title = cleaned
                else:
                    bullets.append(cleaned)
            elif line.startswith('- '):
                bullets.append(line[2:].strip())
            elif line.startswith('* '):
                bullets.append(line[2:].strip())
            elif line.startswith('Slide') and '—' in line:
                # keep as bullet
                bullets.append(line)
            else:
                # plain text, append
                bullets.append(line)

        if not title:
            title = 'Slide'
        # limit bullets to reasonable amount
        add_slide(prs, title, bullets[:10])

    prs.save(pptx_path)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python scripts/generate_pptx_from_md.py input.md output.pptx')
        sys.exit(1)
    md_path = sys.argv[1]
    pptx_path = sys.argv[2]
    make_pptx(md_path, pptx_path)
    print(f'Wrote {pptx_path}')
