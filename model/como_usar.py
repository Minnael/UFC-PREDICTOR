"""
UFC Fight Predictor — Como Usar
================================
Passos:
  1. Coloque este arquivo na mesma pasta que predict.py e a pasta artifacts/
  2. Instale dependências (só na primeira vez):
         pip install torch scikit-learn numpy
  3. Edite os blocos FIGHTER_A e FIGHTER_B com os dados do confronto
  4. Execute:
         python como_usar.py

Campos:
  - Se não souber um valor de performance (sig_strikes_per_min etc.),
    coloque None — o modelo usará a média do conjunto de treino.
  - Campos físicos (height_cm, weight_kg etc.) nunca deixe None.

-----------------------------------------------------------------
LUTA DE EXEMPLO INCLUÍDA:
  Leon Edwards vs Carlos Prates — UFC 309 (15/11/2025)
  Esta luta pertence ao conjunto de TESTE (nov/2025 a mai/2026).
  O modelo foi treinado apenas com lutas até jan/2025,
  portanto NUNCA viu este resultado durante o treinamento.
  Resultado real: Carlos Prates venceu por KO/TKO no R1.
-----------------------------------------------------------------
"""

from predict import FightPredictor


# ══════════════════════════════════════════════════════════════════════════════
#  PREENCHA AQUI — Lutador A
# ══════════════════════════════════════════════════════════════════════════════

FIGHTER_A_NOME = "Leon Edwards"

FIGHTER_A = {
    "height_cm":    188.0,
    "weight_kg":    77.1,
    "reach_arm_cm": 188.0,
    "age":          34.2,
    "stance":       "Southpaw",

    "wins":         14,
    "losses":       4,
    "draws":        0,
    "wins_ko":      3,
    "wins_sub":     1,
    "wins_dec":     10,
    "total_fights": 19,

    "sig_strikes_hit_total": 861,
    "td_hit_total":          26,
    "sub_att_total":         8,
    "kd_total":              5,
    "sig_strikes_per_min":   2.616,
    "td_per_15min":          1.185,
    "sub_per_15min":         0.365,
    "avg_fight_time_sec":    1039.4,
}

# ══════════════════════════════════════════════════════════════════════════════
#  PREENCHA AQUI — Lutador B
# ══════════════════════════════════════════════════════════════════════════════

FIGHTER_B_NOME = "Carlos Prates"

FIGHTER_B = {
    "height_cm":    185.4,
    "weight_kg":    77.1,
    "reach_arm_cm": 198.1,
    "age":          32.2,
    "stance":       "Switch",

    "wins":         6,
    "losses":       1,
    "draws":        0,
    "wins_ko":      6,
    "wins_sub":     0,
    "wins_dec":     0,
    "total_fights": 7,

    "sig_strikes_hit_total": 244,
    "td_hit_total":          1,
    "sub_att_total":         0,
    "kd_total":              9,
    "sig_strikes_per_min":   3.817,
    "td_per_15min":          0.235,
    "sub_per_15min":         0.0,
    "avg_fight_time_sec":    547.9,
}

# ══════════════════════════════════════════════════════════════════════════════
#  NÃO MEXA ABAIXO
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    predictor = FightPredictor("artifacts")
    result    = predictor.predict(
        FIGHTER_A, FIGHTER_B,
        name_a=FIGHTER_A_NOME,
        name_b=FIGHTER_B_NOME,
    )
    predictor.print_prediction(result)
