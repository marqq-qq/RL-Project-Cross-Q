"""Rebuild the editable slides and the presenter script from the experiment exports."""
import argparse
import csv
import json
import io
import zipfile
import re
from lxml import etree
from pathlib import Path

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_TICK_LABEL_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
PAPER = "https://proceedings.iclr.cc/paper_files/paper/2024/hash/f381114cf5aba4e45552869863deaaa7-Abstract-Conference.html"
GYM = "https://github.com/Farama-Foundation/Gymnasium/blob/v1.2.2/gymnasium/envs/box2d/lunar_lander.py"
DOC = "https://sb3-contrib.readthedocs.io/en/master/modules/crossq.html"
IMPL = "https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/v2.7.1/sb3_contrib/crossq/crossq.py"
ZOO = "https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/crossq.yml"
SAC = "https://proceedings.mlr.press/v80/haarnoja18b.html"
NAVY, WHITE, MINT, MUTED, BLUE, ORANGE, INK = "101D2D", "F4F6F5", "92E5D8", "B5C6D1", "087A81", "AD5128", "192D3B"
FONT = "Arial"
I = Inches
def rgb(s): return RGBColor.from_string(s)
def load(p): return json.loads(p.read_text())
def read(p):
    with p.open() as f: return list(csv.DictReader(f))


def text(slide, s, x, y, w, h, size=22, color=WHITE, bold=False, font=FONT, align=PP_ALIGN.LEFT):
    shape = slide.shapes.add_textbox(I(x), I(y), I(w), I(h))
    tf = shape.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for i, line in enumerate(s.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_before = p.space_after = Pt(0)
        p.alignment = align
        run = p.add_run()
        run.text = line
        run.font.name, run.font.size, run.font.bold, run.font.color.rgb = font, Pt(size), bold, rgb(color)
    return shape


def slide(deck, title=None, light=False):
    s = deck.slides.add_slide(deck.slide_layouts[6])
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = rgb("F8FAF8" if light else NAVY)
    if title:
        text(s, title, .65, .48, 12.05, .78, 35, INK if light else WHITE, True)
    return s


def arrow(s, x, y, w=.50):
    sh = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, I(x), I(y), I(w), I(.22))
    sh.fill.solid()
    sh.fill.fore_color.rgb = rgb(MINT)
    sh.line.fill.background()


def img(s, p, x, y, w):
    s.shapes.add_picture(str(p), I(x), I(y), width=I(w))


def stack(s, x, y, title, body, w=5.6):
    text(s, title, x, y, w, .5, 23, MINT, True)
    text(s, body, x, y+.55, w, 1.03, 20, WHITE)


def table(s, values, widths, x, y, height):
    t = s.shapes.add_table(len(values), len(widths), I(x), I(y), I(sum(widths)), I(height)).table
    for col, width in zip(t.columns, widths):
        col.width = I(width)
    for i, row in enumerate(values):
        for j, value in enumerate(row):
            c = t.cell(i, j)
            c.fill.solid()
            c.fill.fore_color.rgb = rgb(INK if i == 0 else ("E6F1F0" if i%2 else "F8FAF8"))
            c.margin_left, c.margin_right = I(.12), I(.06)
            c.margin_top = c.margin_bottom = 0
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = c.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if j==0 else PP_ALIGN.RIGHT
            r = p.add_run()
            r.text = str(value)
            r.font.name, r.font.size = FONT, Pt(17)
            r.font.bold, r.font.color.rgb = (i==0 or j==2), rgb(WHITE if i==0 else INK)
    return t


def sync_chart_cache(filename):
    """Match cached chart values to the exact decimal spelling Excel stores in its sheet."""
    with zipfile.ZipFile(filename) as archive:
        files = {k:archive.read(k) for k in archive.namelist()}
    chart_path = "ppt/charts/chart1.xml"
    chart = etree.fromstring(files[chart_path])
    ns = {"c":chart.nsmap["c"]}
    embedded = next(name for name in files if name.endswith(".xlsx"))
    with zipfile.ZipFile(io.BytesIO(files[embedded])) as workbook:
        sheet = etree.fromstring(workbook.read("xl/worksheets/sheet1.xml"))
    cells = {cell.get("r"):cell.findtext("{*}v") for cell in sheet.findall(".//{*}c")}
    for series in chart.findall(".//c:ser",ns):
        ref = series.find("c:val/c:numRef",ns)
        formula = ref.findtext("c:f",namespaces=ns)
        col, first = re.search(r"!\$([A-Z]+)\$(\d+):", formula).groups()
        for pt in ref.findall("c:numCache/c:pt",ns):
            pt.find("c:v",ns).text = cells[f"{col}{int(first)+int(pt.get('idx'))}"]
    files[chart_path] = etree.tostring(chart,encoding="UTF-8",xml_declaration=True,standalone=True)
    with zipfile.ZipFile(filename,"w",zipfile.ZIP_DEFLATED) as archive:
        for name,contents in files.items():
            archive.writestr(name,contents)


def build(stills, output):
    results = ROOT / "results"
    tr, ev, video = (load(p) for p in [results/"training_summary.json", results/"evaluation_results.json", ROOT/"videos/demo.json"])
    h = read(results/"training_progress.csv")
    baseline, learned = ev["policies"]["random"], ev["policies"]["crossq"]
    n, win = learned["n"], learned["counts"]["landed"]
    selected, seed = tr["selected_step"], video["episodes"][0]["seed"]
    assert tr["model_sha256"] == ev["model_sha256"] == video["model_sha256"]
    prs = Presentation()
    prs.slide_width, prs.slide_height = 12192000, 6858000
    prs.core_properties.title = "CrossQ for Continuous Lunar Landing"
    prs.core_properties.author = "Team 19"
    script = ["# CrossQ for Continuous Lunar Landing", "Team 19 · Speaker script · 4 minutes 30 seconds of introductory slides. Play the separate demo after slide 5, then open the live notebook."]
    def note(s, title, duration, content, sources=()):
        refs = "\n".join(sources)
        s.notes_slide.notes_text_frame.text = content + ("\n\nSources\n" + refs if refs else "")
        script.extend([f"## {len(prs.slides)} · {title} ({duration})", content])
        if refs: script.append("\n".join(f"- {r}" for r in sources))

    s = slide(prs)
    text(s, "Team 19   /   Reinforcement Learning", .7, .74, 8, .42, 18, MINT)
    text(s, "CrossQ for\ncontinuous\nlunar landing", .65, 1.7, 6.68, 3.05, 48, WHITE, True)
    img(s, stills/f"crossq_{seed}_approach.png", 7.26, 1.68, 5.45)
    text(s, "A learned two-engine policy in Gymnasium", .7, 5.43, 11.6, .49, 24, MUTED)
    text(s, "Danil Khasanshin  ·  Arseniy Laptev  ·  Malik Nurullin\nMartin Klimov  ·  Nikita Selezenev", .7, 6.23, 11.9, .81, 17, WHITE)
    note(s, "Danil Khasanshin · The question", "0:40", "Look at the lander above the two flags. It can earn points by moving closer, but it still needs to slow its descent, keep both feet underneath it and eventually cut thrust. A policy that learns only the first part can hover until time runs out. Our question was whether CrossQ could learn the whole sequence using continuous engine commands. We will introduce the experiment here. After the flights, each of us will walk through one part of the same notebook.")

    s=slide(prs, "What the pilot sees and controls")
    img(s, stills/f"crossq_{seed+1}_approach.png", .66, 1.64, 5.64)
    text(s, "A seeded initial push changes the approach.", .67, 5.65, 5.64, .82, 19, MUTED)
    stack(s, 6.85, 1.52, "Eight observations", "Position (x, y), velocity (vₓ, vᵧ), angle, angular velocity and two leg contacts.", 5.75)
    stack(s, 6.85, 3.23, "Main engine", "Off at m ≤ 0. Positive commands give (1 + m) / 2 power.", 5.75)
    stack(s, 6.85, 4.93, "Side boosters", "Left below −0.5, right above +0.5. Power equals |u| outside the dead zone.", 5.75)
    note(s, "Danil Khasanshin · Inputs and engines", "0:50", "The picture lets us watch the approach. The actor receives eight numbers. Horizontal velocity tells it how fast it is drifting, angle tells it which way thrust will point, and the last two numbers tell it when each leg touches the terrain. The two controls have dead zones. For example, a main command of 0.6 requests 80 percent power. A lateral command of minus 0.75 fires the left booster at 75 percent. Approaching upright and slowly improves the shaping score, while engine use costs reward. Martin will explain how the agent learns which commands are worth applying.", [GYM])

    s=slide(prs, "One critic pass gives both sides of the TD target")
    stack(s,.7,1.52,"Current rows","Replay state s with its recorded action a.",3.82)
    stack(s,.7,3.69,"Successor rows","Replay state s′ with a fresh actor sample a′.",3.82)
    arrow(s,4.48,3.26,.55)
    text(s,"Joint batch",5.24,2.21,3.15,.65,28,MINT,True)
    text(s,"256 rows through each online critic. Normalization sees both distributions.",5.24,3.04,3.12,1.99,21)
    arrow(s,8.53,3.26,.55)
    stack(s,9.3,1.52,"First 128","Fit both current Q estimates to the target.",3.27)
    stack(s,9.3,3.69,"Last 128","Take the smaller next Q and detach the target.",3.23)
    text(s,"CrossQ uses the online critics for bootstrapping and keeps their target values fixed during the update.",.7,6.02,12,.85,22,MUTED)
    note(s, "Martin Klimov · The joint critic pass", "0:55", "SAC gives us a stochastic actor and two critics that estimate the soft return of each action. Consider one replay sample. The action actually taken belongs on the current side of the value update. On the successor side, the actor proposes a fresh action. Those two inputs can have different feature distributions. CrossQ puts both sets of rows through each critic together, so normalization measures both on one scale. We split the outputs and hold the target fixed for the gradient step. Our live example will show the difference between separate and joint statistics. Malik will then use PyTorch to take one critic step we can check by hand.", [PAPER, IMPL])

    s=slide(prs,"Training and checkpoint selection",True)
    text(s,"Mean validation return on ten seeds",.70,1.4,7.8,.43,18,INK)
    data=CategoryChartData()
    data.categories=[str(int(row["step"])//1000) for row in h]
    data.add_series("Standard CrossQ validation",[round(float(row["validation_mean"]),8) for row in h])
    chart=s.shapes.add_chart(XL_CHART_TYPE.LINE,I(.72),I(1.96),I(7.68),I(3.88),data).chart
    chart.has_legend=False
    chart.font.name,chart.font.size=FONT,Pt(14)
    chart.category_axis.tick_label_position=XL_TICK_LABEL_POSITION.LOW
    chart.category_axis.tick_labels.font.size=Pt(12)
    chart.value_axis.has_major_gridlines=True
    chart.value_axis.major_gridlines.format.line.color.rgb=rgb("D2DEDC")
    chart.series[0].format.line.color.rgb=rgb(BLUE)
    chart.series[0].format.line.width=Pt(3)
    chart.series[0].smooth=False
    text(s,"Environment steps (thousands), standard run",1.04,5.93,7.2,.46,17,INK)
    text(s,"3 × 200,000",8.74,1.73,3.90,.83,35,BLUE,True)
    text(s,"steps across three fresh runs, each initialized with seed 19",8.78,2.65,3.85,1.3,20,INK)
    text(s,f"{selected:,}",8.74,4.07,3.9,.84,35,BLUE,True)
    text(s,f"steps at the selected checkpoint, with a validation mean of {tr['selected_validation_mean']:.2f}",8.78,4.99,3.87,1.47,20,INK)
    text(s,"Freeze the winner. Test deterministic CrossQ and uniform random actions on the same 50 new seeds.",.72,6.57,12.0,.67,18,INK)
    note(s,"Arseniy Laptev · Training and selection","1:05",f"Each run began afresh with a ten-thousand-step random warm-up. The sparse pilot made one critic update every four interactions and often hovered. The standard run increased that to one update per interaction. A third run kept the sparse update rate and charged one extra point per training step. All three used the original rewards in validation. Follow the line for the standard run. Its validation return stayed near zero for a long stretch before rising sharply. We evaluated every ten thousand steps on the same ten validation seeds. The highest mean across sixty checkpoints was {tr['selected_validation_mean']:.2f} at {selected:,} steps of the standard run. Only then did we compare the frozen actor with uniform random actions on fifty new seeds.",[DOC,ZOO])

    s=slide(prs,f"CrossQ landed on {win} of {n} held-out seeds",True)
    text(s,"Same terrain and initial push for both policies on each seed.",.7,1.37,12,.51,21,BLUE)
    fields=[("Mean return","mean_reward"),("Sample SD","sd_reward"),("Median return","median_reward"),("Mean length (steps)","mean_length")]
    values=[["Measure","Random","CrossQ"]]+[[a,f"{baseline[k]:.1f}",f"{learned[k]:.1f}"] for a,k in fields]
    for label,k in [("Landings","landed"),("Hull crashes","crash"),("Horizontal exits","out_of_bounds"),("Time limits","time_limit")]:
        values.append([label,str(baseline["counts"][k]),str(learned["counts"][k])])
    table(s,values,[3.62,1.65,1.68],.73,2.08,4.29)
    text(s,f"{ev['mean_paired_gain']:+.1f}",8.24,2.49,4.3,1.12,48,BLUE,True)
    text(s,"mean return gain per seed",8.27,3.56,4.36,.91,22,INK)
    lo,hi=ev["paired_bootstrap_95_ci"]
    text(s,f"95% paired bootstrap interval\n{lo:+.1f} to {hi:+.1f}",8.27,4.86,4.36,1.31,20,INK)
    text(s,"Watch the three stages of flight, then open the live notebook.",.73,6.68,12.0,.48,19,BLUE,True)
    note(s,"Nikita Selezenev · Test flights and handoff","1:00",f"Every row uses the same fifty initial conditions for both policies. The selected actor averaged {learned['mean_reward']:.2f} points, compared with {baseline['mean_reward']:.2f} for uniform random actions. Look at the outcomes as well as the mean. CrossQ landed on {win} seeds and timed out twice. The random policy crashed thirty times and left the horizontal boundary twenty times. We use the simulator's sleeping-body condition to identify a landing. The video now shows separate example seeds. Look for the difference between the intermediate actor hovering and the selected actor settling. After the end card, Danil will open the notebook and show exactly what the pilot observes.")
    prs.save(output)
    sync_chart_cache(output)
    (ROOT/"presentation_script.md").write_text("\n\n".join(script)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stills", type=Path, default=ROOT/"presentation/stills")
    p.add_argument("--output", type=Path, default=ROOT/"Final_Presentation.pptx")
    a=p.parse_args()
    build(a.stills,a.output)
