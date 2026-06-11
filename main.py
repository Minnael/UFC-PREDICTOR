from model.predict import FightPredictor
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import json
import copy
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
    alignment=TA_CENTER, spaceAfter=4)

s_method = ParagraphStyle("method",
    fontSize=8.5, fontName="Helvetica-Oblique",
    textColor=colors.HexColor("#444444"),
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
    if conf_pct < 10: return "#888888" # Cinza (Incerta)
    if conf_pct < 20: return "#E6A817" # Amarelo (Moderada)
    if conf_pct < 30: return "#E07B1A" # Laranja (Alta)
    return "#C8102E"               # Vermelho (Muito Alta)

# ── Função para gerar os relatórios dinamicamente ─────────────
def gerar_relatorio(evento_nome, considerar_idade=True):
    sufixo = "- Oficial" if considerar_idade else "- Sem Influencia Idade"
    nome_arquivo = f"{evento_nome} {sufixo}.pdf"
    
    doc = SimpleDocTemplate(
        nome_arquivo,
        pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
    )

    story = []

    # Cabeçalho adaptado
    story.append(Paragraph("UFC FIGHT PREDICTOR", s_title))
    if considerar_idade:
        story.append(Paragraph(f"Gerado em {datetime.now().strftime('%d/%m/%Y %H:%M')}", s_subtitle))
    else:
        aviso_idade = ParagraphStyle("aviso", parent=s_subtitle, textColor=colors.HexColor("#E07B1A"))
        story.append(Paragraph(f"Simulação What-If: Ignorando a Vantagem de Idade<br/>Gerado em {datetime.now().strftime('%d/%m/%Y %H:%M')}", aviso_idade))
        
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#C8102E"), spaceAfter=20))

    for i, fight in enumerate(fights, 1):
        nomes = list(fight.keys())
        nome_a, nome_b = nomes[0], nomes[1]
        
        stats_a = copy.deepcopy(fight[nome_a])
        stats_b = copy.deepcopy(fight[nome_b])

        # Cenário hipotético: iguala as idades se a flag for False
        if not considerar_idade:
            if "age" in stats_a and "age" in stats_b:
                idade_media = (stats_a["age"] + stats_b["age"]) / 2.0
                stats_a["age"] = idade_media
                stats_b["age"] = idade_media

        # Executa a predição multitarefa
        result = predictor.predict(stats_a, stats_b, name_a=nome_a, name_b=nome_b)
        prob_a = result["prob_a"]
        prob_b = result["prob_b"]
        conf   = result["confidence"]
        fav    = result["favorite"]
        clabel = result["conf_label"].split(" ", 1)[-1].strip()
        
        # Coleta das novas probabilidades de método de vitória
        metodos = result["method_probs"]
        txt_metodo = (f"<b>Método Estimado:</b> Nocaute/TKO ({metodos['Nocaute']}%)  |  "
                      f"Finalização ({metodos['Finalização']}%)  |  Decisão ({metodos['Decisão']}%)")

        # Estrutura visual da luta no PDF
        story.append(Paragraph(f"LUTA {i} DE {len(fights)}", s_fight_num))
        story.append(Paragraph(f"{nome_a}  vs  {nome_b}", s_vs))
        story.append(prob_bar_table(nome_a, prob_a, nome_b, prob_b))
        story.append(Spacer(1, 8))

        cc = conf_color(conf)
        story.append(Paragraph(f'Favorito: <font color="{cc}"><b>{fav}</b></font>', s_favorite))
        story.append(Paragraph(f"Confiança: {clabel} ({conf:.1f}%)", s_conf))
        story.append(Paragraph(txt_metodo, s_method))

        if i < len(fights):
            story.append(HRFlowable(width="100%", thickness=0.5,
                                    color=colors.HexColor("#DDDDDD"), spaceAfter=14))

    # Rodapé institucional
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#C8102E"), spaceBefore=10))
    msg_rodape = "Modelo treinado com dados históricos do ufcstats.com — apenas para fins informativos."
    if not considerar_idade:
         msg_rodape = "CENÁRIO HIPOTÉTICO: Idades de ambos os lutadores igualadas para remover viés de juventude. " + msg_rodape
         
    story.append(Paragraph(
        msg_rodape,
        ParagraphStyle("footer", fontSize=7, textColor=colors.HexColor("#AAAAAA"), alignment=TA_CENTER)
    ))

    def add_metadata(canvas, doc):
        canvas.setAuthor("Minnael Campelo")
        canvas.setTitle(f"{evento_nome} {sufixo}")
        canvas.setSubject("Previsões de lutas do UFC")
        canvas.setCreator("UFC Fight Predictor")

    doc.build(story, onFirstPage=add_metadata, onLaterPages=add_metadata)
    print(f"✅ PDF salvo: '{nome_arquivo}'")

# ── Execução ──────────────────────────────────────────────────
evento_alvo = "UFC Freedom 250"

print("Gerando relatórios automatizados...")
# 1. Relatório oficial considerando todas as variáveis físicas
gerar_relatorio(evento_alvo, considerar_idade=True)

# 2. Relatório simulado neutralizando a vantagem de idade
gerar_relatorio(evento_alvo, considerar_idade=False)

print("Processo finalizado com sucesso!")