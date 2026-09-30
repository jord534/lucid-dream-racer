"""Render the technical note to PDF, with its figures.

    python reports/build_pdf.py

Writes reports/technical_note.html always, and reports/technical_note.pdf if WeasyPrint
is installed. Without it, open the HTML in a browser and print to PDF.
"""
from pathlib import Path

import markdown

HERE = Path(__file__).resolve().parent
CSS = """
@page { size: A4; margin: 20mm 18mm;
  @bottom-center { content: counter(page); font: 9pt Helvetica, sans-serif; color: #666 } }
body { font: 10.5pt/1.5 Georgia, 'Times New Roman', serif; color: #111; max-width: 46em;
       margin: 0 auto; padding: 0 1em }
h1 { font: bold 19pt Helvetica, sans-serif; line-height: 1.25; margin: 0 0 .3em }
h2 { font: bold 13pt Helvetica, sans-serif; margin: 1.6em 0 .5em;
     border-bottom: 1px solid #ccc; padding-bottom: .2em; page-break-after: avoid }
h3 { font: bold 11pt Helvetica, sans-serif; margin: 1.2em 0 .4em; page-break-after: avoid }
p, li { orphans: 3; widows: 3 }
table { border-collapse: collapse; width: 100%; margin: .8em 0; font: 9.5pt Helvetica, sans-serif;
        page-break-inside: avoid }
th { background: #eee; text-align: left; border-bottom: 1.5px solid #999 }
th, td { padding: 4px 7px; border-bottom: .5px solid #ddd; vertical-align: top }
img { max-width: 100%; margin: .6em 0; page-break-inside: avoid }
code { font: 9pt 'DejaVu Sans Mono', monospace; background: #f2f2f2; padding: 0 3px }
pre { background: #f6f6f6; padding: .7em; font: 9pt 'DejaVu Sans Mono', monospace;
      page-break-inside: avoid }
hr { border: 0; border-top: .5px solid #bbb; margin: 1.6em 0 }
blockquote { color: #444; border-left: 3px solid #ccc; margin: 0; padding-left: 1em }
"""


def main():
    md = (HERE / "technical_note.md").read_text()
    missing = [ln.split("(")[-1].rstrip(")") for ln in md.splitlines()
               if ln.startswith("![") and not (HERE / ln.split("(")[-1].rstrip(")")).exists()]
    for m in missing:
        print(f"missing figure: {m}  (run: python diagnostics/make_figures.py)")
    body = markdown.markdown(md, extensions=["tables", "fenced_code", "sane_lists"])
    html = (f'<!DOCTYPE html><html><head><meta charset="utf-8">'
            f'<title>Technical note</title><style>{CSS}</style></head>'
            f'<body>{body}</body></html>')
    out_html = HERE / "technical_note.html"
    out_html.write_text(html)
    print(f"wrote {out_html}")
    try:
        from weasyprint import HTML
    except Exception:
        print("WeasyPrint not installed: open the HTML in a browser and print to PDF.")
        print("  (to install: brew install pango gdk-pixbuf libffi && pip install weasyprint)")
        return
    HTML(string=html, base_url=str(HERE)).write_pdf(HERE / "technical_note.pdf")
    print(f"wrote {HERE / 'technical_note.pdf'}")


if __name__ == "__main__":
    main()
