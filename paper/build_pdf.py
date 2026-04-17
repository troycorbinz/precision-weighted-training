"""
Build a publication-quality PDF of the paper from Markdown.

Pipeline:
  Markdown --(pandoc)--> HTML with MathJax + arXiv-style CSS
  HTML --(playwright headless Chromium)--> PDF

Dependencies (one-time setup):
  pip install pypandoc-binary playwright
  python -m playwright install chromium

Run:
  python paper/build_pdf.py

Output:
  paper/precision-weighted-training.pdf  (~600 KB, 27 pages)
"""
import asyncio
from pathlib import Path
import pypandoc
from playwright.async_api import async_playwright

HERE = Path(__file__).parent
MD = HERE / "precision-weighted-training.md"
HTML = HERE / "_build_paper.html"
PDF = HERE / "precision-weighted-training.pdf"

CSS = """
@page {
  size: letter;
  margin: 1in 1in 1in 1in;
  @bottom-center {
    content: counter(page) " / " counter(pages);
    font-family: 'Charter', 'Georgia', serif;
    font-size: 9pt;
    color: #666;
  }
}
html { font-size: 11pt; }
body {
  font-family: 'Charter', 'Georgia', 'Times New Roman', serif;
  line-height: 1.45;
  color: #111;
  max-width: none;
  margin: 0;
  padding: 0;
  text-align: justify;
  hyphens: auto;
}
h1 {
  font-size: 18pt;
  line-height: 1.2;
  margin-top: 0;
  margin-bottom: 0.6em;
  text-align: left;
  font-weight: 700;
}
h2 {
  font-size: 13pt;
  margin-top: 1.6em;
  margin-bottom: 0.4em;
  font-weight: 700;
  text-align: left;
  border-bottom: 1px solid #ccc;
  padding-bottom: 0.15em;
}
h3 {
  font-size: 11.5pt;
  margin-top: 1.2em;
  margin-bottom: 0.25em;
  font-weight: 700;
  text-align: left;
}
h4 {
  font-size: 11pt;
  margin-top: 1em;
  margin-bottom: 0.2em;
  font-weight: 700;
  font-style: italic;
  text-align: left;
}
p { margin: 0.4em 0; orphans: 3; widows: 3; }
strong { font-weight: 700; }
em { font-style: italic; }
code {
  font-family: 'Consolas', 'Courier New', monospace;
  font-size: 9.5pt;
  background: #f4f4f4;
  padding: 1px 3px;
  border-radius: 2px;
}
pre {
  font-family: 'Consolas', 'Courier New', monospace;
  font-size: 9pt;
  background: #f7f7f7;
  border: 1px solid #e2e2e2;
  border-radius: 3px;
  padding: 8px 10px;
  overflow-x: auto;
  line-height: 1.3;
  page-break-inside: avoid;
}
pre code { background: transparent; padding: 0; }
table {
  border-collapse: collapse;
  margin: 0.8em auto;
  font-size: 9.5pt;
  page-break-inside: avoid;
}
th, td {
  border: 1px solid #bbb;
  padding: 4px 8px;
  text-align: left;
  vertical-align: top;
}
th {
  background: #f0f0f0;
  font-weight: 700;
}
blockquote {
  border-left: 3px solid #888;
  margin: 0.6em 0;
  padding: 0 0 0 1em;
  color: #444;
  font-style: italic;
}
img {
  max-width: 100%;
  display: block;
  margin: 0.8em auto;
  page-break-inside: avoid;
}
figure { page-break-inside: avoid; margin: 0.8em 0; text-align: center; }
figcaption { font-size: 9.5pt; color: #444; margin-top: 0.3em; }
a { color: #0033aa; text-decoration: none; }
a:hover { text-decoration: underline; }
hr { border: none; border-top: 1px solid #ccc; margin: 1.5em 0; }
ul, ol { margin: 0.4em 0 0.4em 1.6em; padding: 0; }
li { margin: 0.15em 0; }
/* MathJax math blocks */
.MathJax_Display, mjx-container[display="true"] {
  page-break-inside: avoid;
  margin: 0.6em 0 !important;
}
/* First-page header styling (author block) */
body > p:first-of-type,
body > p:nth-of-type(2),
body > p:nth-of-type(3),
body > p:nth-of-type(4) {
  text-align: left;
  margin: 0.15em 0;
}
/* Section-6/7 subsection titles readability */
h3 + p { margin-top: 0.2em; }
"""

MATHJAX = """
<script>
window.MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
    displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
    processEscapes: true,
  },
  svg: { fontCache: 'global' },
  startup: {
    ready: () => {
      MathJax.startup.defaultReady();
      MathJax.startup.promise.then(() => {
        window.mathJaxDone = true;
      });
    }
  }
};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
"""


def build_html():
    print("Converting markdown to HTML via pandoc...")
    body = pypandoc.convert_file(
        str(MD),
        "html5",
        format="markdown-implicit_figures",
        extra_args=[
            "--wrap=none",
            "--mathjax",
            "--syntax-highlighting=none",
        ],
    )
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Precision-Weighted Training for Language Models</title>
{MATHJAX}
<style>{CSS}</style>
</head>
<body>
{body}
</body>
</html>
"""
    HTML.write_text(html, encoding="utf-8")
    print(f"  HTML: {HTML}")


async def build_pdf():
    print("Rendering HTML to PDF via Chromium...")
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto(HTML.resolve().as_uri())
        # Wait for MathJax to finish typesetting
        await page.wait_for_function("window.mathJaxDone === true", timeout=30000)
        # Also wait for images to settle
        await page.wait_for_load_state("networkidle")
        await page.pdf(
            path=str(PDF),
            format="Letter",
            margin={"top": "0.85in", "right": "0.85in", "bottom": "0.85in", "left": "0.85in"},
            print_background=True,
            prefer_css_page_size=True,
            display_header_footer=False,
        )
        await browser.close()
    print(f"  PDF: {PDF}")


if __name__ == "__main__":
    build_html()
    asyncio.run(build_pdf())
    print(f"\nDone. Output size: {PDF.stat().st_size / 1024:.0f} KB")
