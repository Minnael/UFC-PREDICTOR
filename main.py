from model.predict_method import FullPredictor
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                                 HRFlowable, KeepTogether, PageBreak)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.graphics.shapes import Drawing, Rect
import json
import copy
import os
from datetime import datetime

# ── Dados ─────────────────────────────────────────────────────
with open(os.path.join("scripts", "next_event_stats.json"), encoding="utf-8") as f:
    fights = json.load(f)

predictor = FullPredictor(win_artifacts="model/artifacts", method_artifacts="model/artifacts_method")

# ── Paleta institucional ────────────────────────────────────────
# Cada cor existe em duas formas: string hexadecimal (para uso dentro de
# tags <font color="..."> em Paragraphs) e objeto Color (para tabelas,
# linhas e formas gráficas).
HEX_PRIMARIA      = "#8C1D2B"   # bordo / vermelho institucional
HEX_SECUNDARIA    = "#1B2A4A"   # azul-marinho
HEX_TEXTO         = "#1A1A1A"   # quase preto
HEX_TEXTO_MEDIO   = "#4A4A4A"
HEX_TEXTO_CLARO   = "#6E6E6E"
HEX_DESTAQUE      = "#B8860B"   # dourado discreto, para valores-chave
HEX_ALERTA        = "#B36A00"

COR_PRIMARIA      = colors.HexColor(HEX_PRIMARIA)
COR_SECUNDARIA    = colors.HexColor(HEX_SECUNDARIA)
COR_TEXTO         = colors.HexColor(HEX_TEXTO)
COR_TEXTO_MEDIO   = colors.HexColor(HEX_TEXTO_MEDIO)
COR_TEXTO_CLARO   = colors.HexColor(HEX_TEXTO_CLARO)
COR_LINHA         = colors.HexColor("#CCCCCC")
COR_FUNDO_ALT     = colors.HexColor("#F4F4F6")
COR_CABECALHO_TAB = colors.HexColor("#1B2A4A")
COR_DESTAQUE      = colors.HexColor(HEX_DESTAQUE)
COR_ALERTA        = colors.HexColor(HEX_ALERTA)

NOME_AUTOR = "Minnael Campelo"

# ── Estilos ───────────────────────────────────────────────────
s_title = ParagraphStyle("title",
    fontSize=21, fontName="Helvetica-Bold",
    textColor=COR_PRIMARIA,
    alignment=TA_CENTER, spaceAfter=2, leading=25)

s_subtitle = ParagraphStyle("subtitle",
    fontSize=11, fontName="Helvetica",
    textColor=COR_TEXTO_MEDIO,
    alignment=TA_CENTER, spaceBefore=4, spaceAfter=2, leading=15)

s_meta = ParagraphStyle("meta",
    fontSize=9, fontName="Helvetica",
    textColor=COR_TEXTO_CLARO,
    alignment=TA_CENTER, spaceBefore=0, spaceAfter=18)

s_fight_num = ParagraphStyle("fight_num",
    fontSize=9, fontName="Helvetica-Bold",
    textColor=COR_SECUNDARIA,
    alignment=TA_CENTER, spaceAfter=1, tracking=0.5)

s_weight_class = ParagraphStyle("weight_class",
    fontSize=8.5, fontName="Helvetica-Oblique",
    textColor=COR_TEXTO_CLARO,
    alignment=TA_CENTER, spaceAfter=8)

s_vs = ParagraphStyle("vs",
    fontSize=15, fontName="Helvetica-Bold",
    textColor=COR_TEXTO,
    alignment=TA_CENTER, spaceAfter=10)

s_favorite = ParagraphStyle("favorite",
    fontSize=10, fontName="Helvetica-Bold",
    textColor=COR_TEXTO,
    alignment=TA_CENTER, spaceAfter=0)

s_conf = ParagraphStyle("conf",
    fontSize=9, fontName="Helvetica",
    textColor=COR_TEXTO_MEDIO,
    alignment=TA_CENTER, spaceAfter=6)

s_odds = ParagraphStyle("odds",
    fontSize=9, fontName="Helvetica",
    textColor=COR_TEXTO_MEDIO,
    alignment=TA_CENTER, spaceAfter=8)

s_no_pred = ParagraphStyle("no_pred",
    fontSize=12, fontName="Helvetica-Bold",
    textColor=COR_TEXTO_MEDIO,
    alignment=TA_CENTER, spaceBefore=10, spaceAfter=4)

s_no_pred_reason = ParagraphStyle("no_pred_reason",
    fontSize=8.5, fontName="Helvetica",
    textColor=COR_TEXTO_CLARO,
    alignment=TA_CENTER, spaceAfter=14)

s_section_title = ParagraphStyle("section_title",
    fontSize=16, fontName="Helvetica-Bold",
    textColor=COR_PRIMARIA,
    alignment=TA_CENTER, spaceAfter=6, spaceBefore=10)

s_section_intro = ParagraphStyle("section_intro",
    fontSize=9.5, fontName="Helvetica",
    textColor=COR_TEXTO_MEDIO,
    alignment=TA_CENTER, spaceBefore=4, spaceAfter=16, leading=13)

s_footer_disclaimer = ParagraphStyle("footer_disc",
    fontSize=7.5, fontName="Helvetica",
    textColor=COR_TEXTO_CLARO,
    alignment=TA_CENTER, leading=10)


# ── Mapeamento de categoria de peso ─────────────────────────────
WEIGHT_CLASSES = [
    (53.0, "Peso Palha (Strawweight)"),
    (58.0, "Peso Mosca (Flyweight)"),
    (62.0, "Peso Galo (Bantamweight)"),
    (67.0, "Peso Pena (Featherweight)"),
    (71.0, "Peso Leve (Lightweight)"),
    (78.0, "Meio-Médio (Welterweight)"),
    (85.0, "Peso Médio (Middleweight)"),
    (94.0, "Meio-Pesado (Light Heavyweight)"),
]

def get_weight_class(weight_kg):
    if not weight_kg or weight_kg <= 0:
        return "Categoria não informada"
    for limite, nome in WEIGHT_CLASSES:
        if weight_kg <= limite:
            return nome
    return "Peso Pesado (Heavyweight)"


# ── Helper: barra grafica de probabilidade ──────────────────────
def prob_bar_drawing(prob_pct, bar_color, width=260, height=11):
    d = Drawing(width, height)
    d.add(Rect(0, 0, width, height, fillColor=COR_FUNDO_ALT, strokeColor=COR_LINHA, strokeWidth=0.5))
    fill_w = max(0.0, min(width, width * (prob_pct / 100.0)))
    if fill_w > 0:
        d.add(Rect(0, 0, fill_w, height, fillColor=bar_color, strokeColor=None))
    return d

def prob_bar_table(name_a, prob_a, name_b, prob_b):
    col_name = ParagraphStyle("cn", fontSize=9, fontName="Helvetica-Bold", textColor=COR_TEXTO)
    col_pct  = ParagraphStyle("cp", fontSize=9.5, fontName="Helvetica-Bold", textColor=COR_TEXTO, alignment=TA_CENTER)

    data = [
        [Paragraph(name_a, col_name), prob_bar_drawing(prob_a, COR_PRIMARIA), Paragraph(f"{prob_a:.1f}%", col_pct)],
        [Paragraph(name_b, col_name), prob_bar_drawing(prob_b, COR_SECUNDARIA), Paragraph(f"{prob_b:.1f}%", col_pct)],
    ]
    t = Table(data, colWidths=[5*cm, 9*cm, 2.2*cm])
    t.setStyle(TableStyle([
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",    (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ("LEFTPADDING",   (0,0), (-1,-1), 0),
        ("LINEBELOW",     (0,0), (-1,-2), 0.4, COR_LINHA),
    ]))
    return t

def conf_color(conf_pct):
    """Retorna a cor (em hexadecimal) associada ao nível de confiança da previsão."""
    if conf_pct < 10: return HEX_TEXTO_CLARO
    if conf_pct < 20: return HEX_DESTAQUE
    if conf_pct < 35: return HEX_ALERTA
    return HEX_PRIMARIA

def table_header_style(n_cols):
    return TableStyle([
        ("ALIGN",         (0,0), (-1,-1), "CENTER"),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ("BACKGROUND",    (0,0), (-1,0), COR_CABECALHO_TAB),
        ("TEXTCOLOR",     (0,0), (-1,0), colors.white),
        ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
        ("GRID",          (0,0), (-1,-1), 0.4, COR_LINHA),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [colors.white, COR_FUNDO_ALT]),
        ("TOPPADDING",    (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
    ])

def method_table(outcomes, name_a, name_b):
    col_hdr = ParagraphStyle("ch", fontSize=8.5, fontName="Helvetica-Bold", alignment=TA_CENTER, textColor=colors.white)
    col_txt = ParagraphStyle("ct", fontSize=8.5, fontName="Helvetica", alignment=TA_CENTER, textColor=COR_TEXTO)

    data = [
        [Paragraph("Método de Vitória", col_hdr), Paragraph(name_a, col_hdr), Paragraph(name_b, col_hdr)],
        [Paragraph("Nocaute (KO/TKO)", col_txt), Paragraph(f"{outcomes.get(f'{name_a} — KO/TKO', 0):.1f}%", col_txt), Paragraph(f"{outcomes.get(f'{name_b} — KO/TKO', 0):.1f}%", col_txt)],
        [Paragraph("Finalização (Submissão)", col_txt), Paragraph(f"{outcomes.get(f'{name_a} — Submissão', 0):.1f}%", col_txt), Paragraph(f"{outcomes.get(f'{name_b} — Submissão', 0):.1f}%", col_txt)],
        [Paragraph("Decisão dos Juízes", col_txt), Paragraph(f"{outcomes.get(f'{name_a} — Decisão', 0):.1f}%", col_txt), Paragraph(f"{outcomes.get(f'{name_b} — Decisão', 0):.1f}%", col_txt)],
    ]
    t = Table(data, colWidths=[4.5*cm, 4.75*cm, 4.75*cm])
    t.setStyle(table_header_style(3))
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
    col_hdr   = ParagraphStyle("ch", fontSize=8.5, fontName="Helvetica-Bold", alignment=TA_CENTER, textColor=colors.white)
    col_txt   = ParagraphStyle("ct", fontSize=8.5, fontName="Helvetica-Bold", alignment=TA_CENTER, textColor=COR_DESTAQUE)
    col_label = ParagraphStyle("cl", fontSize=8.5, fontName="Helvetica", alignment=TA_CENTER, textColor=COR_TEXTO)
    col_calc  = ParagraphStyle("cc", fontSize=7.5, fontName="Helvetica", alignment=TA_CENTER, textColor=COR_TEXTO_CLARO)

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

    if odd_u25 > 99.99: odd_u25 = 99.99
    if odd_u15 > 99.99: odd_u15 = 99.99
    if odd_o25 > 99.99: odd_o25 = 99.99
    if odd_o15 > 99.99: odd_o15 = 99.99

    data = [
        [Paragraph("Linha (Assaltos)", col_hdr), Paragraph("Cálculo da Probabilidade Real", col_hdr), Paragraph("Odd Justa (EV+)", col_hdr)],
        [Paragraph("Menos de 2.5", col_label), Paragraph(f"({prob_via_rapida:.1f}% × {fator_25:.2f}) = <b>{prob_u25:.1f}%</b>", col_calc), Paragraph(f"{odd_u25:.2f}", col_txt)],
        [Paragraph("Mais de 2.5", col_label), Paragraph(f"(100% − {prob_u25:.1f}%) = <b>{prob_o25:.1f}%</b>", col_calc), Paragraph(f"{odd_o25:.2f}", col_txt)],
        [Paragraph("Menos de 1.5", col_label), Paragraph(f"({prob_via_rapida:.1f}% × {fator_15:.2f}) = <b>{prob_u15:.1f}%</b>", col_calc), Paragraph(f"{odd_u15:.2f}", col_txt)],
        [Paragraph("Mais de 1.5", col_label), Paragraph(f"(100% − {prob_u15:.1f}%) = <b>{prob_o15:.1f}%</b>", col_calc), Paragraph(f"{odd_o15:.2f}", col_txt)],
    ]

    t = Table(data, colWidths=[3.5*cm, 7*cm, 3.5*cm])
    t.setStyle(table_header_style(3))
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
        leftMargin=2.2*cm, rightMargin=2.2*cm,
        topMargin=2.6*cm, bottomMargin=2.3*cm,
    )

    story = []

    story.append(Paragraph("UFC FIGHT PREDICTOR", s_title))
    story.append(Paragraph("Relatório Técnico de Análise Preditiva de Combates", s_subtitle))
    if considerar_idade:
        story.append(Paragraph(
            f"Evento: {evento_nome}  |  Gerado em {datetime.now().strftime('%d/%m/%Y às %H:%M')}",
            s_meta))
    else:
        aviso_idade = ParagraphStyle("aviso", parent=s_meta, textColor=COR_ALERTA, fontName="Helvetica-Bold")
        story.append(Paragraph(
            f"CENÁRIO HIPOTÉTICO — Vantagem de idade neutralizada entre os atletas<br/>"
            f"Evento: {evento_nome}  |  Gerado em {datetime.now().strftime('%d/%m/%Y às %H:%M')}",
            aviso_idade))

    story.append(HRFlowable(width="100%", thickness=1.2, color=COR_PRIMARIA, spaceAfter=20))

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
        if fights_a == 0: tag_a = f'<font color="{HEX_PRIMARIA}" size="7">[ESTREANTE]  </font>'
        elif fights_a <= 3: tag_a = f'<font color="{HEX_ALERTA}" size="7">[{fights_a} LUTAS]  </font>'

        tag_b = ""
        if fights_b == 0: tag_b = f'<font color="{HEX_PRIMARIA}" size="7">  [ESTREANTE]</font>'
        elif fights_b <= 3: tag_b = f'<font color="{HEX_ALERTA}" size="7">  [{fights_b} LUTAS]</font>'

        display_a = f"{tag_a}{nome_a}"
        display_b = f"{nome_b}{tag_b}"

        peso_kg = stats_a.get("weight_kg", 0) or stats_b.get("weight_kg", 0)
        categoria_peso = get_weight_class(peso_kg)

        fight_elements = []
        fight_elements.append(Paragraph(f"LUTA {i} DE {len(fights)}", s_fight_num))
        fight_elements.append(Paragraph(f"Categoria de Peso: {categoria_peso}", s_weight_class))
        fight_elements.append(Paragraph(f"{display_a}  vs  {display_b}", s_vs))

        if "error" in result:
            fight_elements.append(Paragraph("PREVISÃO INDISPONÍVEL", s_no_pred))
            fight_elements.append(Paragraph(
                "O modelo exige histórico mínimo de uma luta no UFC para ambos os atletas a fim de "
                "emitir uma previsão estatisticamente fundamentada.",
                s_no_pred_reason))
        else:
            prob_a = result["prob_a"]
            prob_b = result["prob_b"]
            conf   = result["confidence"]
            fav    = result["favorite"]
            clabel = result["conf_label"].split(" ", 1)[-1].strip()

            fight_elements.append(prob_bar_table(nome_a, prob_a, nome_b, prob_b))

            cc = conf_color(conf)
            fight_elements.append(Paragraph(
                f'Favorito: <font color="{cc}"><b>{fav}</b></font>', s_favorite))
            fight_elements.append(Paragraph(f"Nível de Confiança: {clabel} ({conf:.1f}%)", s_conf))

            fav_prob = prob_a if fav == nome_a else prob_b
            fav_odd = 100.0 / fav_prob if fav_prob > 0 else 99.99
            underdog_prob = prob_b if fav == nome_a else prob_a
            underdog_odd = 100.0 / underdog_prob if underdog_prob > 0 else 99.99

            fight_elements.append(Paragraph(
                f"Odd Justa — Favorito: <b>{fav_odd:.2f}</b>  |  Azarão: <b>{underdog_odd:.2f}</b>", s_odds))

            fight_elements.append(Spacer(1, 4))
            fight_elements.append(method_table(result["outcomes"], nome_a, nome_b))
            fight_elements.append(Spacer(1, 6))
            fight_elements.append(time_odds_table(result["outcomes"], nome_a, nome_b, peso_kg))

        fight_elements.append(Spacer(1, 10))

        if i < len(fights):
            fight_elements.append(HRFlowable(width="100%", thickness=0.5,
                                    color=COR_LINHA, spaceAfter=10))

        story.append(KeepTogether(fight_elements))

    # Rodapé da última página de lutas
    story.append(HRFlowable(width="100%", thickness=1.2, color=COR_PRIMARIA, spaceBefore=8, spaceAfter=8))
    msg_rodape = ("Relatório gerado por modelo estatístico treinado com dados históricos do UFCSTATS.com. "
                  "Documento de caráter exclusivamente informativo, sem qualquer garantia de resultado.")
    if not considerar_idade:
        msg_rodape = ("CENÁRIO HIPOTÉTICO: as idades de ambos os atletas foram igualadas para remover o "
                       "viés de juventude da análise. " + msg_rodape)

    story.append(Paragraph(msg_rodape, s_footer_disclaimer))

    # ── Página anexa: metodologia dos fatores de tempo ──────────
    story.append(PageBreak())
    story.append(Paragraph("Anexo Metodológico", s_section_title))
    story.append(Paragraph("Fatores de Desconto de Tempo por Categoria de Peso", s_subtitle))
    story.append(HRFlowable(width="40%", thickness=1, color=COR_PRIMARIA, spaceBefore=6, spaceAfter=16, hAlign="CENTER"))

    col_d_hdr = ParagraphStyle("cdh", fontSize=9.5, fontName="Helvetica-Bold", alignment=TA_CENTER, textColor=colors.white)
    col_d_txt = ParagraphStyle("cdt", fontSize=9, fontName="Helvetica", alignment=TA_CENTER, textColor=COR_TEXTO)
    col_d_hl  = ParagraphStyle("cdhl", fontSize=9, fontName="Helvetica-Bold", alignment=TA_CENTER, textColor=COR_DESTAQUE)

    data_desc = [
        [Paragraph("Categoria de Peso", col_d_hdr), Paragraph("Fator Under 2.5", col_d_hdr), Paragraph("Fator Under 1.5", col_d_hdr)],
        [Paragraph("Peso Palha (52 kg)", col_d_txt), Paragraph("0.81", col_d_hl), Paragraph("0.42", col_d_hl)],
        [Paragraph("Peso Mosca (57 kg)", col_d_txt), Paragraph("0.91", col_d_hl), Paragraph("0.59", col_d_hl)],
        [Paragraph("Peso Galo (61 kg)", col_d_txt), Paragraph("0.86", col_d_hl), Paragraph("0.65", col_d_hl)],
        [Paragraph("Peso Pena (65 kg)", col_d_txt), Paragraph("0.92", col_d_hl), Paragraph("0.67", col_d_hl)],
        [Paragraph("Peso Leve (70 kg)", col_d_txt), Paragraph("0.92", col_d_hl), Paragraph("0.67", col_d_hl)],
        [Paragraph("Meio-Médio (77 kg)", col_d_txt), Paragraph("0.90", col_d_hl), Paragraph("0.68", col_d_hl)],
        [Paragraph("Peso Médio (84 kg)", col_d_txt), Paragraph("0.91", col_d_hl), Paragraph("0.66", col_d_hl)],
        [Paragraph("Meio-Pesado (93 kg)", col_d_txt), Paragraph("0.94", col_d_hl), Paragraph("0.72", col_d_hl)],
        [Paragraph("Peso Pesado (120 kg)", col_d_txt), Paragraph("0.90", col_d_hl), Paragraph("0.70", col_d_hl)],
    ]

    t_desc = Table(data_desc, colWidths=[6*cm, 4*cm, 4*cm])
    t_desc.setStyle(table_header_style(3))
    story.append(t_desc)

    story.append(Paragraph(
        "Esta tabela apresenta a taxa histórica de conversão do UFC: entre todas as lutas que não chegam "
        "à decisão dos juízes, qual proporção se encerra antes dos minutos finais de cada assalto. "
        "As <b>Odds Justas (EV+)</b> apresentadas ao longo deste relatório são calculadas multiplicando "
        "a probabilidade de desfecho por via rápida por estes fatores.",
        s_section_intro))

    def add_page_furniture(canvas, doc):
        canvas.saveState()
        # Régua superior
        canvas.setStrokeColor(COR_LINHA)
        canvas.setLineWidth(0.5)
        canvas.line(2.2*cm, A4[1] - 1.6*cm, A4[0] - 2.2*cm, A4[1] - 1.6*cm)
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(COR_TEXTO_CLARO)
        canvas.drawString(2.2*cm, A4[1] - 1.35*cm, "UFC FIGHT PREDICTOR")
        canvas.drawRightString(A4[0] - 2.2*cm, A4[1] - 1.35*cm, evento_nome)

        # Rodapé
        canvas.setStrokeColor(COR_LINHA)
        canvas.line(2.2*cm, 1.6*cm, A4[0] - 2.2*cm, 1.6*cm)
        canvas.setFont("Helvetica", 7.5)
        canvas.drawString(2.2*cm, 1.25*cm, f"Gerado em {datetime.now().strftime('%d/%m/%Y')}")
        canvas.drawCentredString(A4[0] / 2, 1.25*cm, "Documento informativo — não constitui aconselhamento de aposta")
        canvas.drawRightString(A4[0] - 2.2*cm, 1.25*cm, f"Página {doc.page}")

        canvas.setAuthor(NOME_AUTOR)
        canvas.setTitle(f"{evento_nome} {sufixo}")
        canvas.setSubject("Previsões de lutas do UFC")
        canvas.setCreator("UFC Fight Predictor")
        canvas.restoreState()

    doc.build(story, onFirstPage=add_page_furniture, onLaterPages=add_page_furniture)
    print(f"PDF salvo: '{nome_arquivo}'")

# ── Execução ──────────────────────────────────────────────────
evento_alvo = input("Digite o nome do evento: ")

print("Gerando relatórios...")
# 1. Gera o PDF normal (com as idades reais)
gerar_relatorio(evento_alvo, considerar_idade=True)

# 2. Gera o PDF hipotético (onde ambos têm a mesma idade)
gerar_relatorio(evento_alvo, considerar_idade=False)