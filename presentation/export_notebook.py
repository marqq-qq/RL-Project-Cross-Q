"""Export the executed notebook as a self-contained, offline HTML reading copy."""
from pathlib import Path
from html import escape

import nbformat
from nbconvert import HTMLExporter
from pygments.formatters import HtmlFormatter

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "CrossQ_LunarLander_Final.ipynb"
STYLES = """
:root {color-scheme:light; font:19px/1.48 Arial, sans-serif; color:#182d3a; background:#fafcfb}
body {margin:0 auto; max-width:1040px; padding:32px clamp(16px,5vw,64px) 90px}
h1,h2,h3 {line-height:1.18; text-wrap:balance} h1 {font-size:2.4rem}
h2 {color:#096f75; margin:2.25rem 0 .7rem; padding-top:.65rem; border-top:1px solid #b9cfce}
h3 {font-size:1.22rem; margin:1.7rem 0 .5rem} p {max-width:90ch}
table {border-collapse:collapse; margin:.75rem 0; font-size:.96rem; max-width:100%}
td,th {text-align:left; padding:.38rem .72rem; border-bottom:1px solid #c7d7d8}
thead {background:#e2efed} pre {overflow-x:auto; font: .85rem/1.48 ui-monospace, Menlo, monospace; margin:.3rem 0; padding:.6rem .8rem}
.input_area,.input {background:#eff4f4} .output_area pre {border-left:3px solid #09828a; background:#fff}
.output_area,.rendered_html {overflow:auto} .cell {margin:.5rem 0} .prompt,.anchor-link {display:none}
img {max-width:100%; height:auto; display:block; margin:.6rem 0}
a {color:#096f75} code {font-size:.89em} .output_scroll {height:auto; max-height:none}
@media print {body{font-size:12pt} table,pre,img{break-inside:avoid}}
"""
def export():
    book = nbformat.read(NOTEBOOK, as_version=4)
    if any(c.cell_type == "code" and c.execution_count is None for c in book.cells):
        raise ValueError("Execute the notebook before exporting the reading copy.")
    renderer = HTMLExporter(template_name="basic", exclude_input_prompt=True, exclude_output_prompt=True)
    body, _ = renderer.from_notebook_node(book)
    syntax = HtmlFormatter().get_style_defs(".highlight")
    title = escape("CrossQ for continuous lunar landing · Team 19")
    html = (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title>'
            f'<style>{STYLES}{syntax}</style></head><body><main>{body}</main></body></html>')
    NOTEBOOK.with_suffix(".html").write_text(html,encoding="utf-8")


if __name__=="__main__":
    export()
