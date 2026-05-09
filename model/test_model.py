"""
UFC Fight Predictor — Como Usar
================================
Passos:
  1. Coloque este arquivo na mesma pasta que predict.py e a pasta artifacts/
  2. Instale dependências (só na primeira vez):
         pip install torch scikit-learn numpy
  3. Edite os blocos FIGHTER_A e FIGHTER_B com os dados do confronto
  4. Execute:
         python main.py

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

FIGHTER_A_NOME = "Khamzat Chimaev"

FIGHTER_A = {
    "height_cm": 188.0     ,
    "weight_kg": 83.9      ,
    "reach_arm_cm": 190.5     ,
    "stance": "Orthodox",
    "age": 32.0      ,
    "wins": 9         ,
    "losses": 0         ,
    "draws": 0         ,
    "wins_ko": 2         ,
    "wins_sub": 4         ,
    "wins_dec": 3         ,
    "total_fights": 9         ,
    "sig_strikes_hit_total": 298.0     ,
    "td_hit_total": 26.0      ,
    "sub_att_total": 9.0       ,
    "kd_total": 2.0       ,
    "sig_strikes_per_min": 4.044     ,
    "td_per_15min": 5.293     ,
    "sub_per_15min": 1.832     ,
    "avg_fight_time_sec": 491.2     ,
}

# ══════════════════════════════════════════════════════════════════════════════
#  PREENCHA AQUI — Lutador B
# ══════════════════════════════════════════════════════════════════════════════

FIGHTER_B_NOME = "Sean Strickland"

FIGHTER_B = {
    "height_cm": 185.4     ,
    "weight_kg": 83.9      ,
    "reach_arm_cm": 193.0     ,
    "stance": "Orthodox",
    "age": 35.2      ,
    "wins": 17        ,
    "losses": 7         ,
    "draws": 0         ,
    "wins_ko": 5         ,
    "wins_sub": 1         ,
    "wins_dec": 11        ,
    "total_fights": 24        ,
    "sig_strikes_hit_total": 2307.0    ,
    "td_hit_total": 18.0      ,
    "sub_att_total": 4.0       ,
    "kd_total": 7.0       ,
    "sig_strikes_per_min": 6.045     ,
    "td_per_15min": 0.707     ,
    "sub_per_15min": 0.157     ,
    "avg_fight_time_sec": 954.1     ,
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
