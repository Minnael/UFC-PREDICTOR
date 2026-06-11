from model.predict import FightPredictor
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import json
from datetime import datetime

# ── Dados ─────────────────────────────────────────────────────
with open("next_event_stats.json", encoding="utf-8") as f:
    fights = json.load(f)

predictor = FightPredictor("model/artifacts")

# ── Estilos ───────────────────────────────────────────────────
s_title = ParagraphStyle("title",
    fontSize=22, fontName="Helvetica-Bold",
    textColor=colors.HexColor("#C8102E"),
    alignment=TA_CENTER, spaceAfter=4)

s_subtitle = ParagraphStyle("subtitle",
    fontSize=11, fontName="Helvetica",
    textColor=colors.HexColor("#666666"),
    alignment=TA_CENTER, spaceBefore=20, spaceAfter=20)

s_fight_num = ParagraphStyle("fight_num",
    fontSize=9, fontName="Helvetica",
    textColor=colors.HexColor("#999999"),
    alignment=TA_CENTER, spaceAfter=2)

s_vs = ParagraphStyle("vs",
    fontSize=16, fontName="Helvetica-Bold",
    textColor=colors.HexColor("#1A1A1A"),
    alignment=TA_CENTER, spaceAfter=6)

s_favorite = ParagraphStyle("favorite",
    fontSize=10, fontName="Helvetica-Bold",
    textColor=colors.HexColor("#C8102E"),
    alignment=TA_CENTER, spaceAfter=2)

s_conf = ParagraphStyle("conf",
    fontSize=9, fontName="Helvetica",
    textColor=colors.HexColor("#555555"),
    alignment=TA_CENTER, spaceAfter=16)

# ── Helper: barra de probabilidade ────────────────────────────
def prob_bar_table(name_a, prob_a, name_b, prob_b):
    bar_total = 30
    fill_a = int(round(prob_a / 100 * bar_total))
    fill_b = bar_total - fill_a

    bar_a = "\u2588" * fill_a + "\u2591" * (bar_total - fill_a)
    bar_b = "\u2591" * (bar_total - fill_b) + "\u2588" * fill_b

    col_name  = ParagraphStyle("cn",  fontSize=9,  fontName="Helvetica-Bold", textColor=colors.HexColor("#1A1A1A"))
    col_bar_a = ParagraphStyle("cba", fontSize=8,  fontName="Helvetica",      textColor=colors.HexColor("#C8102E"))
    col_bar_b = ParagraphStyle("cbb", fontSize=8,  fontName="Helvetica",      textColor=colors.HexColor("#1560BD"))
    col_pct   = ParagraphStyle("cp",  fontSize=10, fontName="Helvetica-Bold", textColor=colors.HexColor("#333333"), alignment=TA_CENTER)

    data = [
        [Paragraph(name_a, col_name), Paragraph(bar_a, col_bar_a), Paragraph(f"{prob_a:.1f}%", col_pct)],
        [Paragraph(name_b, col_name), Paragraph(bar_b, col_bar_b), Paragraph(f"{prob_b:.1f}%", col_pct)],
    ]
    t = Table(data, colWidths=[5*cm, 9*cm, 2.5*cm])
    t.setStyle(TableStyle([
        ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",   (0,0), (-1,-1), 5),
        ("BOTTOMPADDING",(0,0), (-1,-1), 5),
        ("LEFTPADDING",  (0,0), (-1,-1), 0),
        ("ROWBACKGROUNDS",(0,0),(-1,-1), [colors.HexColor("#F9F9F9"), colors.white]),
        ("LINEBELOW",    (0,0), (-1,-2), 0.3, colors.HexColor("#DDDDDD")),
    ]))
    return t

def conf_color(conf_pct):
    if conf_pct < 10: return "#888888"
    if conf_pct < 20: return "#E6A817"
    if conf_pct < 35: return "#E07B1A"
    return "#C8102E"


# ── Monta PDF ─────────────────────────────────────────────────
doc = SimpleDocTemplate(
    "ufc_predictions.pdf",
    pagesize=A4,
    leftMargin=2*cm, rightMargin=2*cm,
    topMargin=2*cm, bottomMargin=2*cm,
)

story = []

story.append(Paragraph("UFC FIGHT PREDICTOR", s_title))
story.append(Paragraph(f"Gerado em {datetime.now().strftime('%d/%m/%Y %H:%M')}", s_subtitle))
story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#C8102E"), spaceAfter=20))

for i, fight in enumerate(fights, 1):
    nomes  = list(fight.keys())
    nome_a, nome_b = nomes[0], nomes[1]
    stats_a, stats_b = fight[nome_a], fight[nome_b]

    result = predictor.predict(stats_a, stats_b, name_a=nome_a, name_b=nome_b)
    prob_a = result["prob_a"]
    prob_b = result["prob_b"]
    conf   = result["confidence"]
    fav    = result["favorite"]
    clabel = result["conf_label"].split(" ", 1)[-1].strip()

    story.append(Paragraph(f"LUTA {i} DE {len(fights)}", s_fight_num))
    story.append(Paragraph(f"{nome_a}  vs  {nome_b}", s_vs))
    story.append(prob_bar_table(nome_a, prob_a, nome_b, prob_b))
    story.append(Spacer(1, 8))

    cc = conf_color(conf)
    story.append(Paragraph(f'Favorito: <font color="{cc}"><b>{fav}</b></font>', s_favorite))
    story.append(Paragraph(f"Confianca: {clabel} ({conf:.1f}%)", s_conf))

    if i < len(fights):
        story.append(HRFlowable(width="100%", thickness=0.5,
                                color=colors.HexColor("#DDDDDD"), spaceAfter=14))

story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#C8102E"), spaceBefore=10))
story.append(Paragraph(
    "Modelo treinado com dados historicos do ufcstats.com — apenas para fins informativos.",
    ParagraphStyle("footer", fontSize=7, textColor=colors.HexColor("#AAAAAA"), alignment=TA_CENTER)
))

def add_metadata(canvas, doc):
    canvas.setAuthor("Minnael Campelo")
    canvas.setTitle("UFC FREEDOM -> SEM IDADE")
    canvas.setSubject("Previsões de lutas do UFC")
    canvas.setCreator("UFC Fight Predictor")

doc.build(
    story,
    onFirstPage=add_metadata,
    onLaterPages=add_metadata
)
print("PDF salvo em ufc_predictions.pdf")