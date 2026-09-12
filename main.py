from model.predict_method import FullPredictor
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, KeepTogether, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
import json
import copy
import os
from datetime import datetime

# ── Dados ─────────────────────────────────────────────────────
with open(os.path.join("scripts", "next_event_stats.json"), encoding="utf-8") as f:
    fights = json.load(f)

predictor = FullPredictor(win_artifacts="model/artifacts", method_artifacts="model/artifacts_method")

# ── Estilos ───────────────────────────────────────────────────
s_title = ParagraphStyle("title",
    fontSize=22, fontName="Helvetica-Bold",
    textColor=colors.HexColor("#C8102E"),
    alignment=TA_CENTER, spaceAfter=4)

s_subtitle = ParagraphStyle("subtitle",
    fontSize=11, fontName="Helvetica",
    textColor=colors.HexColor("#AAAAAA"),
    alignment=TA_CENTER, spaceBefore=20, spaceAfter=20)

s_fight_num = ParagraphStyle("fight_num",
    fontSize=9, fontName="Helvetica",
    textColor=colors.HexColor("#777777"),
    alignment=TA_CENTER, spaceAfter=2)

s_vs = ParagraphStyle("vs",
    fontSize=16, fontName="Helvetica-Bold",
    textColor=colors.HexColor("#FFFFFF"),
    alignment=TA_CENTER, spaceAfter=8)

s_favorite = ParagraphStyle("favorite",
    fontSize=10, fontName="Helvetica-Bold",
    textColor=colors.HexColor("#E74C3C"),
    alignment=TA_CENTER, spaceAfter=0)

s_conf = ParagraphStyle("conf",
    fontSize=9, fontName="Helvetica",
    textColor=colors.HexColor("#CCCCCC"),
    alignment=TA_CENTER, spaceAfter=6)

s_no_pred = ParagraphStyle("no_pred",
    fontSize=13, fontName="Helvetica-Bold",
    textColor=colors.HexColor("#888888"),
    alignment=TA_CENTER, spaceBefore=14, spaceAfter=6)

s_no_pred_reason = ParagraphStyle("no_pred_reason",
    fontSize=8, fontName="Helvetica",
    textColor=colors.HexColor("#666666"),
    alignment=TA_CENTER, spaceAfter=14)

# ── Helper: barra de probabilidade ────────────────────────────
def prob_bar_table(name_a, prob_a, name_b, prob_b):
    bar_total = 30
    fill_a = int(round(prob_a / 100 * bar_total))
    fill_b = bar_total - fill_a

    bar_a = "\u2588" * fill_a + "\u2591" * (bar_total - fill_a)
    bar_b = "\u2591" * (bar_total - fill_b) + "\u2588" * fill_b

    col_name  = ParagraphStyle("cn",  fontSize=9,  fontName="Helvetica-Bold", textColor=colors.HexColor("#FFFFFF"))
    col_bar_a = ParagraphStyle("cba", fontSize=8,  fontName="Helvetica",      textColor=colors.HexColor("#E74C3C"))
    col_bar_b = ParagraphStyle("cbb", fontSize=8,  fontName="Helvetica",      textColor=colors.HexColor("#3498DB"))
    col_pct   = ParagraphStyle("cp",  fontSize=10, fontName="Helvetica-Bold", textColor=colors.HexColor("#DDDDDD"), alignment=TA_CENTER)

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
        ("ROWBACKGROUNDS",(0,0),(-1,-1), [colors.HexColor("#2A2A2A"), colors.HexColor("#1E1E1E")]),
        ("LINEBELOW",    (0,0), (-1,-2), 0.3, colors.HexColor("#444444")),
    ]))
    return t

def conf_color(conf_pct):
    if conf_pct < 10: return "#888888"
    if conf_pct < 20: return "#F1C40F"
    if conf_pct < 35: return "#E67E22"
    return "#E74C3C"

def method_table(outcomes, name_a, name_b):
    col_hdr = ParagraphStyle("ch", fontSize=8, fontName="Helvetica-Bold", alignment=TA_CENTER, textColor=colors.HexColor("#FFFFFF"))
    col_txt = ParagraphStyle("ct", fontSize=8, fontName="Helvetica", alignment=TA_CENTER, textColor=colors.HexColor("#DDDDDD"))
    
    data = [
        [Paragraph("Método", col_hdr), Paragraph(name_a, col_hdr), Paragraph(name_b, col_hdr)],
        [Paragraph("KO/TKO", col_txt), Paragraph(f"{outcomes.get(f'{name_a} — KO/TKO', 0):.1f}%", col_txt), Paragraph(f"{outcomes.get(f'{name_b} — KO/TKO', 0):.1f}%", col_txt)],
        [Paragraph("Submissão", col_txt), Paragraph(f"{outcomes.get(f'{name_a} — Submissão', 0):.1f}%", col_txt), Paragraph(f"{outcomes.get(f'{name_b} — Submissão', 0):.1f}%", col_txt)],
        [Paragraph("Decisão", col_txt), Paragraph(f"{outcomes.get(f'{name_a} — Decisão', 0):.1f}%", col_txt), Paragraph(f"{outcomes.get(f'{name_b} — Decisão', 0):.1f}%", col_txt)],
    ]
    t = Table(data, colWidths=[4*cm, 5*cm, 5*cm])
    t.setStyle(TableStyle([
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#333333")),
        ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#444444")),
        ("TOPPADDING", (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]))
    return t

def get_time_factors(weight_kg):
    if not weight_kg or weight_kg <= 0: return 0.90, 0.65
    if weight_kg <= 53.0: return 0.81, 0.42 # Strawweight
    if weight_kg <= 58.0: return 0.91, 0.59 # Flyweight
    if weight_kg <= 62.0: return 0.86, 0.65 # Bantamweight
    if weight_kg <= 67.0: return 0.92, 0.67 # Featherweight
    if weight_kg <= 71.0: return 0.92, 0.67 # Lightweight
    if weight_kg <= 78.0: return 0.90, 0.68 # Welterweight
    if weight_kg <= 85.0: return 0.91, 0.66 # Middleweight
    if weight_kg <= 94.0: return 0.94, 0.72 # Light Heavyweight
    return 0.90, 0.70 # Heavyweight

def time_odds_table(outcomes, name_a, name_b, weight_kg):
    col_hdr = ParagraphStyle("ch", fontSize=8, fontName="Helvetica-Bold", alignment=TA_CENTER, textColor=colors.HexColor("#FFFFFF"))
    col_txt = ParagraphStyle("ct", fontSize=8, fontName="Helvetica", alignment=TA_CENTER, textColor=colors.HexColor("#F1C40F"))
    col_label = ParagraphStyle("cl", fontSize=8, fontName="Helvetica", alignment=TA_CENTER, textColor=colors.HexColor("#DDDDDD"))
    col_calc = ParagraphStyle("cc", fontSize=7, fontName="Helvetica", alignment=TA_CENTER, textColor=colors.HexColor("#AAAAAA"))

    prob_via_rapida = outcomes.get(f'{name_a} — KO/TKO', 0) + outcomes.get(f'{name_a} — Submissão', 0) + \
                      outcomes.get(f'{name_b} — KO/TKO', 0) + outcomes.get(f'{name_b} — Submissão', 0)
    
    fator_25, fator_15 = get_time_factors(weight_kg)
    
    prob_u25 = prob_via_rapida * fator_25
    prob_u15 = prob_via_rapida * fator_15
    
    prob_o25 = 100.0 - prob_u25
    prob_o15 = 100.0 - prob_u15
    
    odd_u25 = 100.0 / prob_u25 if prob_u25 > 0 else 99.99
    odd_u15 = 100.0 / prob_u15 if prob_u15 > 0 else 99.99
    odd_o25 = 100.0 / prob_o25 if prob_o25 > 0 else 99.99
    odd_o15 = 100.0 / prob_o15 if prob_o15 > 0 else 99.99
    
    # Cap odds at 99.99 visually
    if odd_u25 > 99.99: odd_u25 = 99.99
    if odd_u15 > 99.99: odd_u15 = 99.99
    if odd_o25 > 99.99: odd_o25 = 99.99
    if odd_o15 > 99.99: odd_o15 = 99.99

    data = [
        [Paragraph("Linha (Rounds)", col_hdr), Paragraph("Cálculo da Probabilidade Real", col_hdr), Paragraph("Odd Justa (EV+)", col_hdr)],
        [Paragraph("Menos de 2.5", col_label), Paragraph(f"({prob_via_rapida:.1f}% * {fator_25:.2f}) = <b>{prob_u25:.1f}%</b>", col_calc), Paragraph(f"{odd_u25:.2f}", col_txt)],
        [Paragraph("Mais de 2.5", col_label), Paragraph(f"(100% - {prob_u25:.1f}%) = <b>{prob_o25:.1f}%</b>", col_calc), Paragraph(f"{odd_o25:.2f}", col_txt)],
        [Paragraph("Menos de 1.5", col_label), Paragraph(f"({prob_via_rapida:.1f}% * {fator_15:.2f}) = <b>{prob_u15:.1f}%</b>", col_calc), Paragraph(f"{odd_u15:.2f}", col_txt)],
        [Paragraph("Mais de 1.5", col_label), Paragraph(f"(100% - {prob_u15:.1f}%) = <b>{prob_o15:.1f}%</b>", col_calc), Paragraph(f"{odd_o15:.2f}", col_txt)],
    ]
    
    t = Table(data, colWidths=[3.5*cm, 7*cm, 3.5*cm])
    t.setStyle(TableStyle([
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#333333")),
        ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#444444")),
        ("TOPPADDING", (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]))
    return t

# ── Função para gerar os relatórios dinamicamente ─────────────
def gerar_relatorio(evento_nome, considerar_idade=True):
    sufixo = "- Oficial" if considerar_idade else "- Sem Influencia Idade"
    
    os.makedirs("output", exist_ok=True)

    nome_arquivo = os.path.join(
        "output",
        f"{evento_nome} {sufixo}.pdf"
    )
    
    doc = SimpleDocTemplate(
        nome_arquivo,
        pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
    )

    story = []

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

        if not considerar_idade:
            if "age" in stats_a and "age" in stats_b:
                idade_media = (stats_a["age"] + stats_b["age"]) / 2.0
                stats_a["age"] = idade_media
                stats_b["age"] = idade_media

        result = predictor.predict(stats_a, stats_b, name_a=nome_a, name_b=nome_b)

        fights_a = int(float(stats_a.get("total_fights", 0) or 0))
        fights_b = int(float(stats_b.get("total_fights", 0) or 0))
        
        tag_a = ""
        if fights_a == 0: tag_a = "<font color='#FF0000'>(E) </font>"
        elif fights_a <= 3: tag_a = f"<font color='#FF6600'>({fights_a}) </font>"
            
        tag_b = ""
        if fights_b == 0: tag_b = "<font color='#FF0000'> (E)</font>"
        elif fights_b <= 3: tag_b = f"<font color='#FF6600'> ({fights_b})</font>"

        display_a = f"{tag_a}{nome_a}"
        display_b = f"{nome_b}{tag_b}"

        fight_elements = []
        fight_elements.append(Paragraph(f"LUTA {i} DE {len(fights)}", s_fight_num))
        fight_elements.append(Paragraph(f"{display_a}  vs  {display_b}", s_vs))

        if "error" in result:
            # Estreante(s) sem histórico suficiente — modelo não arrisca previsão
            fight_elements.append(Paragraph("⚪ SEM PREVISÃO DISPONÍVEL", s_no_pred))
            fight_elements.append(Paragraph(
                "O modelo exige pelo menos 1 luta no histórico do UFC para ambos os lutadores.",
                s_no_pred_reason))
        else:
            prob_a = result["prob_a"]
            prob_b = result["prob_b"]
            conf   = result["confidence"]
            fav    = result["favorite"]
            clabel = result["conf_label"].split(" ", 1)[-1].strip()

            fight_elements.append(prob_bar_table(nome_a, prob_a, nome_b, prob_b))

            cc = conf_color(conf)
            fight_elements.append(Paragraph(f'Favorito: <font color="{cc}"><b>{fav}</b></font>', s_favorite))
            fight_elements.append(Paragraph(f"Confianca: {clabel} ({conf:.1f}%)", s_conf))

            fav_prob = prob_a if fav == nome_a else prob_b
            fav_odd = 100.0 / fav_prob if fav_prob > 0 else 99.99
            underdog_prob = prob_b if fav == nome_a else prob_a
            underdog_odd = 100.0 / underdog_prob if underdog_prob > 0 else 99.99

            s_odds = ParagraphStyle("odds", fontSize=9, fontName="Helvetica", textColor=colors.HexColor("#F1C40F"), alignment=TA_CENTER, spaceAfter=6)
            fight_elements.append(Paragraph(f"Odd Justa — Favorito: <b>{fav_odd:.2f}</b> | Azarão: <b>{underdog_odd:.2f}</b>", s_odds))

            fight_elements.append(Spacer(1, 4))
            fight_elements.append(method_table(result["outcomes"], nome_a, nome_b))
            fight_elements.append(Spacer(1, 4))
            fight_elements.append(time_odds_table(result["outcomes"], nome_a, nome_b, stats_a.get("weight_kg", 0)))

        fight_elements.append(Spacer(1, 8))

        if i < len(fights):
            fight_elements.append(HRFlowable(width="100%", thickness=0.5,
                                    color=colors.HexColor("#DDDDDD"), spaceAfter=8))
                                    
        story.append(KeepTogether(fight_elements))

    # Rodapé
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#C8102E"), spaceBefore=10))
    msg_rodape = "Modelo treinado com dados historicos do ufcstats.com — apenas para fins informativos."
    if not considerar_idade:
         msg_rodape = "CENÁRIO HIPOTÉTICO: Idades de ambos os lutadores igualadas para remover viés de juventude. " + msg_rodape
         
    story.append(Paragraph(
        msg_rodape,
        ParagraphStyle("footer", fontSize=7, textColor=colors.HexColor("#AAAAAA"), alignment=TA_CENTER)
    ))

    story.append(PageBreak())
    s_dt = ParagraphStyle("dt", fontSize=18, fontName="Helvetica-Bold", textColor=colors.HexColor("#C8102E"), alignment=TA_CENTER, spaceAfter=20, spaceBefore=40)
    story.append(Paragraph("Fatores de Desconto de Tempo por Categoria", s_dt))
    
    col_d_hdr = ParagraphStyle("cdh", fontSize=10, fontName="Helvetica-Bold", alignment=TA_CENTER, textColor=colors.HexColor("#FFFFFF"))
    col_d_txt = ParagraphStyle("cdt", fontSize=9, fontName="Helvetica", alignment=TA_CENTER, textColor=colors.HexColor("#DDDDDD"))
    col_d_hl = ParagraphStyle("cdhl", fontSize=9, fontName="Helvetica-Bold", alignment=TA_CENTER, textColor=colors.HexColor("#F1C40F"))
    
    data_desc = [
        [Paragraph("Categoria", col_d_hdr), Paragraph("Fator Under 2.5", col_d_hdr), Paragraph("Fator Under 1.5", col_d_hdr)],
        [Paragraph("Strawweight (52kg)", col_d_txt), Paragraph("0.81", col_d_hl), Paragraph("0.42", col_d_hl)],
        [Paragraph("Flyweight (57kg)", col_d_txt), Paragraph("0.91", col_d_hl), Paragraph("0.59", col_d_hl)],
        [Paragraph("Bantamweight (61kg)", col_d_txt), Paragraph("0.86", col_d_hl), Paragraph("0.65", col_d_hl)],
        [Paragraph("Featherweight (65kg)", col_d_txt), Paragraph("0.92", col_d_hl), Paragraph("0.67", col_d_hl)],
        [Paragraph("Lightweight (70kg)", col_d_txt), Paragraph("0.92", col_d_hl), Paragraph("0.67", col_d_hl)],
        [Paragraph("Welterweight (77kg)", col_d_txt), Paragraph("0.90", col_d_hl), Paragraph("0.68", col_d_hl)],
        [Paragraph("Middleweight (84kg)", col_d_txt), Paragraph("0.91", col_d_hl), Paragraph("0.66", col_d_hl)],
        [Paragraph("Light Heavyweight (93kg)", col_d_txt), Paragraph("0.94", col_d_hl), Paragraph("0.72", col_d_hl)],
        [Paragraph("Heavyweight (120kg)", col_d_txt), Paragraph("0.90", col_d_hl), Paragraph("0.70", col_d_hl)]
    ]
    
    t_desc = Table(data_desc, colWidths=[6*cm, 4*cm, 4*cm])
    t_desc.setStyle(TableStyle([
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#333333")),
        ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#444444")),
        ("TOPPADDING", (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
    ]))
    story.append(t_desc)
    
    s_desc_txt = ParagraphStyle("desc", fontSize=9, fontName="Helvetica", textColor=colors.HexColor("#AAAAAA"), alignment=TA_CENTER, spaceBefore=15)
    story.append(Paragraph("Esta tabela mostra a taxa histórica de conversão do UFC: de todas as lutas que não chegam à decisão, quantas terminam antes dos minutos finais.<br/>As <b>Odds Justas (EV+)</b> no relatório são calculadas multiplicando a chance de Via Rápida por estes fatores.", s_desc_txt))

    def add_metadata(canvas, doc):
        canvas.saveState()
        canvas.setFillColor(colors.HexColor("#121212"))
        canvas.rect(0, 0, A4[0], A4[1], fill=1)
        canvas.restoreState()
        
        canvas.setAuthor("Minnael Campelo")
        canvas.setTitle(f"{evento_nome} {sufixo}")
        canvas.setSubject("Previsões de lutas do UFC")
        canvas.setCreator("UFC Fight Predictor")

    doc.build(story, onFirstPage=add_metadata, onLaterPages=add_metadata)
    print(f"PDF salvo: '{nome_arquivo}'")

# ── Execução ──────────────────────────────────────────────────
evento_alvo = input("Digite o nome do evento: ")

print("Gerando relatórios...")
# 1. Gera o PDF normal (com as idades reais)
gerar_relatorio(evento_alvo, considerar_idade=True)

# 2. Gera o PDF hipotético (onde ambos têm a mesma idade)
gerar_relatorio(evento_alvo, considerar_idade=False)