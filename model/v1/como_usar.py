"""
UFC Fight Predictor — Como Usar
================================
1. Coloque este arquivo na mesma pasta que os artefatos:
   best_model.pt, temperature.pt, scaler.pkl, col_medians.npy, meta.json

2. Instale as dependências (só na primeira vez):
   pip install torch scikit-learn pandas numpy

3. Edite a seção "PREENCHA AQUI" com os dados dos dois lutadores

4. Execute:
   python como_usar.py
"""

from model.v1.predict import FightPredictor

# ══════════════════════════════════════════════════════
#  PREENCHA AQUI — Dados do Lutador A
# ══════════════════════════════════════════════════════
FIGHTER_A_NOME = "Carlos Prates"

FIGHTER_A = {
    # Físico
    "height_cm":    185.4,      # Altura em cm
    "weight_kg":    77.1,       # Peso em kg
    "reach_arm_cm": 198.1,      # Alcance em cm
    "age":          32.7,       # Idade na data da luta
    "stance":       "Switch",   # "Orthodox", "Southpaw" ou "Switch"

    # Histórico (ANTES desta luta)
    "wins":         7,
    "losses":       1,
    "draws":        0,
    "wins_ko":      7,          # Vitórias por KO/TKO
    "wins_sub":     0,          # Vitórias por Submission
    "wins_dec":     0,          # Vitórias por Decisão
    "total_fights": 8,          # Total de lutas no cartel

    # Stats acumuladas de performance (se não souber, coloque None)
    "sig_strikes_hit_total": 265,   # Total de golpes significativos acertados
    "td_hit_total":          1,     # Total de takedowns
    "sub_att_total":         0,     # Tentativas de finalização
    "kd_total":              10,    # Knockdowns causados
    "sig_strikes_per_min":   3.765, # Golpes significativos por minuto
    "td_per_15min":          0.213, # Takedowns por 15 min
    "sub_per_15min":         0.0,   # Tentativas de sub por 15 min
    "avg_fight_time_sec":    527.9, # Tempo médio de luta em segundos
}

# ══════════════════════════════════════════════════════
#  PREENCHA AQUI — Dados do Lutador B
# ══════════════════════════════════════════════════════
FIGHTER_B_NOME = "Jack Della Maddalena"

FIGHTER_B = {
    # Físico
    "height_cm":    180.3,
    "weight_kg":    77.1,
    "reach_arm_cm": 185.4,
    "age":          29.6,
    "stance":       "Switch",

    # Histórico (ANTES desta luta)
    "wins":         9,
    "losses":       1,
    "draws":        0,
    "wins_ko":      4,
    "wins_sub":     1,
    "wins_dec":     4,
    "total_fights": 10,

    # Stats acumuladas (None = modelo usa a média do dataset)
    "sig_strikes_hit_total": 667,
    "td_hit_total":          1,
    "sub_att_total":         1,
    "kd_total":              6,
    "sig_strikes_per_min":   5.565,
    "td_per_15min":          0.125,
    "sub_per_15min":         0.125,
    "avg_fight_time_sec":    719.1,
}

# ══════════════════════════════════════════════════════
#  NÃO MEXA ABAIXO — Só executa a predição
# ══════════════════════════════════════════════════════
if __name__ == "__main__":
    predictor = FightPredictor("artifacts")   # artefatos na mesma pasta
    result    = predictor.predict(
        FIGHTER_A, FIGHTER_B,
        name_a=FIGHTER_A_NOME,
        name_b=FIGHTER_B_NOME,
    )
    predictor.print_prediction(result)
