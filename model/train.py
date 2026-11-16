"""
UFC Fight Outcome Predictor — train_v3.py
==========================================
Correções aplicadas vs versões anteriores:
  1. Estreantes removidos: lutas onde qualquer lutador tem total_fights == 0
  2. Split TEMPORAL (70/20/10) — sem data leakage entre passado e futuro
  3. Augmentação por espelhamento feita DENTRO de cada split separadamente,
     garantindo que nenhuma luta (nem seu espelho) cruze entre conjuntos
  4. Mediana e Scaler calculados EXCLUSIVAMENTE no conjunto de treino
  5. Conjunto de teste jamais é tocado antes da avaliação final
"""

import os, re, json, pickle, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import binom
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, log_loss, brier_score_loss,
    classification_report,
)
warnings.filterwarnings("ignore")

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DATA_PATH  = "../scripts/global_distribuido/ufc_dataset.csv"
OUT_DIR    = "artifacts"
os.makedirs(OUT_DIR, exist_ok=True)

STANCE_MAP   = {"Orthodox": 0, "Southpaw": 1, "Switch": 2}
NUMERIC_BASE = [
    "height_cm", "weight_kg", "reach_arm_cm", "age",
    "wins", "losses", "draws", "wins_ko", "wins_sub", "wins_dec",
    "losses_ko", "losses_sub", "losses_dec",
    "total_fights",
    "sig_strikes_hit_total", "td_hit_total", "sub_att_total", "kd_total",
    "sig_strikes_per_min", "td_per_15min", "sub_per_15min",
    "kd_absorbed_total", "sig_strikes_absorbed_total", "td_absorbed_total", "sub_absorbed_total",
    "sig_strikes_absorbed_per_min", "td_absorbed_per_15min", "sub_absorbed_per_15min",
    "avg_fight_time_sec",
    "last_fight_1", "last_fight_2", "last_fight_3", "last_fight_4", "last_fight_5", # ← NOVAS VARIÁVEIS
]

# FUNCIONALIDADE NOVA -> STREAKS, RATING ELO/SoS E DECAIMENTO TEMPORAL (EWMA - PONTO 6)
def _extract_cum_snapshot(row, prefix: str = "") -> dict:
    tf = float(row.get(f"{prefix}total_fights", 0) or 0)
    avg_sec = float(row.get(f"{prefix}avg_fight_time_sec", 0) or 0)
    return {
        "total_fights": tf,
        "total_sec": tf * avg_sec,
        "sig_hit": float(row.get(f"{prefix}sig_strikes_hit_total", 0) or 0),
        "sig_abs": float(row.get(f"{prefix}sig_strikes_absorbed_total", 0) or 0),
        "td_hit": float(row.get(f"{prefix}td_hit_total", 0) or 0),
        "td_abs": float(row.get(f"{prefix}td_absorbed_total", 0) or 0),
        "sub_att": float(row.get(f"{prefix}sub_att_total", 0) or 0),
        "sub_abs": float(row.get(f"{prefix}sub_absorbed_total", 0) or 0),
        "wins": float(row.get(f"{prefix}wins", 0) or 0),
    }


def _update_and_get_ewma(past_bouts: list, current_snap: dict, last_snap: dict,
                         last_bout_meta: dict, current_date: pd.Timestamp,
                         half_life_years: float = 2.5):
    """
    Reconstrói as estatísticas por luta (delta entre snapshots acumulados consecutivos)
    e aplica decaimento exponencial no tempo (EWMA: w_i = exp(-lambda * delta_anos))
    com meia-vida de 2.5 anos para capturar a fase recente vs. média da carreira.
    """
    if last_snap is not None and last_bout_meta is not None:
        d_sec = max(15.0, current_snap["total_sec"] - last_snap["total_sec"])
        past_bouts.append({
            "date": last_bout_meta["date"],
            "won": last_bout_meta["won"],
            "sec": d_sec,
            "sig_hit": max(0.0, current_snap["sig_hit"] - last_snap["sig_hit"]),
            "sig_abs": max(0.0, current_snap["sig_abs"] - last_snap["sig_abs"]),
            "td_hit": max(0.0, current_snap["td_hit"] - last_snap["td_hit"]),
            "td_abs": max(0.0, current_snap["td_abs"] - last_snap["td_abs"]),
            "sub_att": max(0.0, current_snap["sub_att"] - last_snap["sub_att"]),
            "sub_abs": max(0.0, current_snap["sub_abs"] - last_snap["sub_abs"]),
        })

    if not past_bouts or current_snap["total_fights"] <= 0:
        return None

    decay_lambda = np.log(2.0) / half_life_years
    w_sum = w_min_sum = w_wins = 0.0
    w_sig_hit = w_sig_abs = w_td_hit = w_td_abs = 0.0

    for b in past_bouts:
        years_ago = max(0.0, (current_date - b["date"]).days / 365.25)
        w = float(np.exp(-decay_lambda * years_ago))
        b_min = max(0.25, b["sec"] / 60.0)
        w_sum += w
        w_min_sum += w * b_min
        w_wins += w * (1.0 if b["won"] else 0.0)
        w_sig_hit += w * b["sig_hit"]
        w_sig_abs += w * b["sig_abs"]
        w_td_hit += w * b["td_hit"]
        w_td_abs += w * b["td_abs"]

    ewma_win_rate = w_wins / max(w_sum, 1e-6)
    ewma_sig_pm = w_sig_hit / max(w_min_sum, 1e-6)
    ewma_sig_abs_pm = w_sig_abs / max(w_min_sum, 1e-6)
    ewma_td_15 = (w_td_hit / max(w_min_sum, 1e-6)) * 15.0
    ewma_td_abs_15 = (w_td_abs / max(w_min_sum, 1e-6)) * 15.0

    tf = max(1.0, current_snap["total_fights"])
    tot_min = max(0.25, current_snap["total_sec"] / 60.0)
    career_win_rate = current_snap["wins"] / tf
    career_sig_diff = (current_snap["sig_hit"] - current_snap["sig_abs"]) / tot_min
    career_td_diff = ((current_snap["td_hit"] - current_snap["td_abs"]) / tot_min) * 15.0

    return {
        "ewma_win_rate": ewma_win_rate,
        "form_win_trend": ewma_win_rate - career_win_rate,
        "form_sig_trend": (ewma_sig_pm - ewma_sig_abs_pm) - career_sig_diff,
        "form_td_trend": (ewma_td_15 - ewma_td_abs_15) - career_td_diff,
        "ewma_sig_diff": ewma_sig_pm - ewma_sig_abs_pm,
    }


def compute_streaks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula cronologicamente (sem data leakage):
      - win_streak e loss_streak
      - Ponto 1: elo, strike_elo, grapple_elo, opp_elo_avg, beaten_elo_avg, elo_momentum
      - Ponto 6: Decaimento Temporal Exponencial (EWMA, meia-vida de 2.5 anos)
        alocado nos 5 slots de fase recente (last_fight_1..5):
          1. ewma_win_rate (taxa de vitória ponderada exponencialmente no tempo)
          2. form_win_trend (fase atual de vitórias vs. média da carreira)
          3. form_sig_trend (evolução recente do diferencial de golpes/min vs. carreira)
          4. form_td_trend (evolução recente do diferencial de quedas/15m vs. carreira)
          5. ewma_sig_diff (diferencial líquido de golpes/min ponderado pela recência)
    """
    if "a_elo" in df.columns:
        return df

    def _streak(history: list) -> tuple[int, int]:
        """Retorna (win_streak, loss_streak) com base no histórico anterior."""
        if not history:
            return 0, 0
        win_streak = loss_streak = 0
        if history[-1]:   # última luta foi vitória
            for won in reversed(history):
                if won: win_streak += 1
                else: break
        else:             # última luta foi derrota
            for won in reversed(history):
                if not won: loss_streak += 1
                else: break
        return win_streak, loss_streak

    fighter_history: dict[str, list[bool]] = {}
    elo_rating: dict[str, float] = {}
    strike_elo: dict[str, float] = {}
    grapple_elo: dict[str, float] = {}
    elo_trace: dict[str, list[float]] = {}
    opp_elos: dict[str, list[float]] = {}
    beaten_elos: dict[str, list[float]] = {}

    past_bouts: dict[str, list[dict]] = {}
    last_snap: dict[str, dict] = {}
    last_bout_meta: dict[str, dict] = {}

    rows = {
        "a_win_streak": [], "a_loss_streak": [],
        "b_win_streak": [], "b_loss_streak": [],
        "a_elo": [], "b_elo": [],
        "a_strike_elo": [], "b_strike_elo": [],
        "a_grapple_elo": [], "b_grapple_elo": [],
        "a_opp_elo_avg": [], "b_opp_elo_avg": [],
        "a_beaten_elo_avg": [], "b_beaten_elo_avg": [],
        "a_elo_momentum": [], "b_elo_momentum": [],
    }
    for p in ("a_", "b_"):
        for i in range(1, 6):
            rows[f"{p}last_fight_{i}"] = []

    for _, row in df.iterrows():
        fa, fb = row["fighter_a"], row["fighter_b"]
        ev_date = row["event_date"]
        ha = fighter_history.get(fa, [])
        hb = fighter_history.get(fb, [])

        wa, la = _streak(ha)
        wb, lb = _streak(hb)

        rows["a_win_streak"].append(wa)
        rows["a_loss_streak"].append(la)
        rows["b_win_streak"].append(wb)
        rows["b_loss_streak"].append(lb)

        # Ratings Elo e Força do Cartel ANTES da luta acontecer
        ra = elo_rating.get(fa, 1500.0)
        rb = elo_rating.get(fb, 1500.0)
        sa_elo = strike_elo.get(fa, 1500.0)
        sb_elo = strike_elo.get(fb, 1500.0)
        ga_elo = grapple_elo.get(fa, 1500.0)
        gb_elo = grapple_elo.get(fb, 1500.0)

        rows["a_elo"].append(ra)
        rows["b_elo"].append(rb)
        rows["a_strike_elo"].append(sa_elo)
        rows["b_strike_elo"].append(sb_elo)
        rows["a_grapple_elo"].append(ga_elo)
        rows["b_grapple_elo"].append(gb_elo)

        opps_a = opp_elos.get(fa, [])
        opps_b = opp_elos.get(fb, [])
        rows["a_opp_elo_avg"].append(float(np.mean(opps_a)) if opps_a else 1500.0)
        rows["b_opp_elo_avg"].append(float(np.mean(opps_b)) if opps_b else 1500.0)

        beat_a = beaten_elos.get(fa, [])
        beat_b = beaten_elos.get(fb, [])
        rows["a_beaten_elo_avg"].append(float(np.mean(beat_a)) if beat_a else 1500.0)
        rows["b_beaten_elo_avg"].append(float(np.mean(beat_b)) if beat_b else 1500.0)

        tr_a = elo_trace.get(fa, [1500.0])
        tr_b = elo_trace.get(fb, [1500.0])
        rows["a_elo_momentum"].append(ra - (tr_a[-3] if len(tr_a) >= 3 else tr_a[0]))
        rows["b_elo_momentum"].append(rb - (tr_b[-3] if len(tr_b) >= 3 else tr_b[0]))

        # Ponto 6: Decaimento Exponencial no Tempo (EWMA, meia-vida = 2.5 anos)
        snap_a = _extract_cum_snapshot(row, "a_")
        snap_b = _extract_cum_snapshot(row, "b_")
        ewma_a = _update_and_get_ewma(
            past_bouts.setdefault(fa, []), snap_a, last_snap.get(fa),
            last_bout_meta.get(fa), ev_date, half_life_years=2.5
        )
        ewma_b = _update_and_get_ewma(
            past_bouts.setdefault(fb, []), snap_b, last_snap.get(fb),
            last_bout_meta.get(fb), ev_date, half_life_years=2.5
        )
        for p, ew in (("a_", ewma_a), ("b_", ewma_b)):
            if ew is not None:
                rows[f"{p}last_fight_1"].append(ew["ewma_win_rate"])
                rows[f"{p}last_fight_2"].append(ew["form_win_trend"])
                rows[f"{p}last_fight_3"].append(ew["form_sig_trend"])
                rows[f"{p}last_fight_4"].append(ew["form_td_trend"])
                rows[f"{p}last_fight_5"].append(ew["ewma_sig_diff"])
            else:
                for i in range(1, 6):
                    rows[f"{p}last_fight_{i}"].append(0.0)

        # Atualiza histórico COM o resultado desta luta
        a_won = bool(row["label"])
        last_snap[fa] = snap_a
        last_snap[fb] = snap_b
        last_bout_meta[fa] = {"date": ev_date, "won": a_won}
        last_bout_meta[fb] = {"date": ev_date, "won": not a_won}
        fighter_history.setdefault(fa, []).append(a_won)
        fighter_history.setdefault(fb, []).append(not a_won)

        opp_elos.setdefault(fa, []).append(rb)
        opp_elos.setdefault(fb, []).append(ra)
        if a_won:
            beaten_elos.setdefault(fa, []).append(rb)
        else:
            beaten_elos.setdefault(fb, []).append(ra)

        method = str(row.get("method", "") or "")
        try:
            rnd = int(float(row.get("round", 3) or 3))
        except Exception:
            rnd = 3

        is_ko = bool(re.search(r"KO|TKO", method, re.I))
        is_sub = bool(re.search(r"Sub", method, re.I))
        is_split = bool(re.search(r"Split", method, re.I))

        if is_ko or is_sub:
            mult = 1.35 if rnd == 1 else (1.20 if rnd == 2 else 1.15)
        elif is_split:
            mult = 0.75
        else:
            mult = 1.0

        ea = 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))
        score_a = 1.0 if a_won else 0.0
        k_a = 36.0 if len(ha) < 4 else 24.0
        k_b = 36.0 if len(hb) < 4 else 24.0

        new_ra = ra + k_a * mult * (score_a - ea)
        new_rb = rb + k_b * mult * ((1.0 - score_a) - (1.0 - ea))
        elo_rating[fa] = new_ra
        elo_rating[fb] = new_rb
        elo_trace.setdefault(fa, [1500.0]).append(new_ra)
        elo_trace.setdefault(fb, [1500.0]).append(new_rb)

        if is_ko:
            esa = 1.0 / (1.0 + 10.0 ** ((sb_elo - sa_elo) / 400.0))
            strike_elo[fa] = sa_elo + 28.0 * (score_a - esa)
            strike_elo[fb] = sb_elo + 28.0 * ((1.0 - score_a) - (1.0 - esa))
        elif is_sub:
            ega = 1.0 / (1.0 + 10.0 ** ((gb_elo - ga_elo) / 400.0))
            grapple_elo[fa] = ga_elo + 28.0 * (score_a - ega)
            grapple_elo[fb] = gb_elo + 28.0 * ((1.0 - score_a) - (1.0 - ega))

    df = df.copy()
    for col, vals in rows.items():
        df[col] = vals

    # Diagnóstico rápido
    print(f"\nStreaks e Ratings Elo/SoS calculados:")
    print(f"  win_streak  — max: {df['a_win_streak'].max()}  "
          f"média: {df['a_win_streak'].mean():.2f}")
    print(f"  loss_streak — max: {df['a_loss_streak'].max()}  "
          f"média: {df['a_loss_streak'].mean():.2f}")
    print(f"  elo_rating  — min: {df['a_elo'].min():.1f}  "
          f"max: {df['a_elo'].max():.1f}  média: {df['a_elo'].mean():.1f}")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# 1. Carregamento e limpeza
# ══════════════════════════════════════════════════════════════════════════════

def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["event_date"] = pd.to_datetime(df["event_date"])
    df = df.sort_values("event_date", kind="mergesort").reset_index(drop=True)

    before = len(df)

    # Calcula streaks, Elo e Força do Cartel no histórico completo ANTES de filtrar estreantes
    df = compute_streaks(df)

    # Remove lutas onde qualquer lutador é estreante (sem histórico)
    mask = (df["a_total_fights"] > 0) & (df["b_total_fights"] > 0)
    df = df[mask].reset_index(drop=True)

    removed = before - len(df)
    print(f"Lutas originais   : {before}")
    print(f"Estreantes removidos: {removed}")
    print(f"Lutas após filtro  : {len(df)}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. Split temporal — sem embaralhamento, ordem cronológica preservada
# ══════════════════════════════════════════════════════════════════════════════

def temporal_split(df: pd.DataFrame):
    df = df[df['event_date'].dt.year >= 2015].reset_index(drop=True)
    df_train = df[df["event_date"].dt.year <= 2022].reset_index(drop=True)
    df_val = df[df["event_date"].dt.year.isin([2023, 2024])].reset_index(drop=True)
    df_test  = df[df["event_date"].dt.year.isin([2025, 2026])].reset_index(drop=True)

    print(f"\nSplit temporal:")
    print(f"  Treino : {len(df_train):>4} lutas  "
          f"[{df_train['event_date'].min().date()} → {df_train['event_date'].max().date()}]")
    print(f"  Val    : {len(df_val):>4} lutas  "
          f"[{df_val['event_date'].min().date()} → {df_val['event_date'].max().date()}]")
    print(f"  Teste  : {len(df_test):>4} lutas  "
          f"[{df_test['event_date'].min().date()} → {df_test['event_date'].max().date()}]")

    # Verificação de sanidade: nenhuma data do teste aparece no treino
    assert df_train["event_date"].max() <= df_val["event_date"].min(), \
        "ERRO: datas do treino e val se sobrepõem!"
    assert df_val["event_date"].max() <= df_test["event_date"].min(), \
        "ERRO: datas de val e teste se sobrepõem!"
    print("  ✅ Sem sobreposição de datas entre splits")

    return df_train, df_val, df_test


# ══════════════════════════════════════════════════════════════════════════════
# 3. Feature engineering
# ══════════════════════════════════════════════════════════════════════════════

def _derived(stats: dict) -> list:
    wins  = float(stats.get("wins",   0) or 0)
    total = float(stats.get("total_fights", 0) or 0)
    ko    = float(stats.get("wins_ko",  0) or 0)
    sub   = float(stats.get("wins_sub", 0) or 0)
    dec   = float(stats.get("wins_dec", 0) or 0)
    losses= float(stats.get("losses",   0) or 0)
    td_15 = float(stats.get("td_per_15min",  0) or 0)
    sub_15= float(stats.get("sub_per_15min", 0) or 0)
    l_ko  = float(stats.get("losses_ko",  0) or 0)
    l_sub = float(stats.get("losses_sub", 0) or 0)

    return [
        wins / total        if total > 0 else 0.5,   # win_rate
        (ko + sub) / wins   if wins  > 0 else 0.0,   # finish_rate
        ko  / wins          if wins  > 0 else 0.0,   # ko_rate
        sub / wins          if wins  > 0 else 0.0,   # sub_rate
        dec / wins          if wins  > 0 else 0.0,   # dec_rate
        losses / total      if total > 0 else 0.5,   # loss_rate
        float(np.log1p(total)),                       # exp_score
        td_15 + sub_15 * 0.5,                         # grapple_score
        (l_ko + l_sub) / total  if total > 0 else 0.0,  # been_finished_rate
        l_ko  / total           if total > 0 else 0.0,  # ko_vulnerability
        l_sub / total           if total > 0 else 0.0,  # sub_vulnerability
    ]


def build_row_features(row_a: dict, row_b: dict, a_win_streak: float=0, a_loss_streak: float=0, 
                       b_win_streak: float=0, b_loss_streak: float=0) -> np.ndarray:
    """Constrói vetor de features dado stats de A e B como dicts."""

    feats = []

    # 1. Deltas numéricos (A − B)
    for col in NUMERIC_BASE:
        va = float(row_a.get(col, 0) or 0)
        vb = float(row_b.get(col, 0) or 0)
        feats.append(va - vb)

    # 2. Absolutos de A, depois B
    for stats in (row_a, row_b):
        for col in NUMERIC_BASE:
            feats.append(float(stats.get(col, 0) or 0))

    # 3. Features derivadas de A, B e deltas
    da = _derived(row_a)
    db = _derived(row_b)
    feats.extend(da)
    feats.extend(db)
    feats.extend([a - b for a, b in zip(da, db)])

    # INJEÇÃO DOS STREAKS (As novas features)
    feats.extend([a_win_streak, a_loss_streak, b_win_streak, b_loss_streak])
    feats.extend([a_win_streak - b_win_streak, a_loss_streak - b_loss_streak])

    # INJEÇÃO DE RATING ELO DINÂMICO E FORÇA DO CARTEL (Strength of Schedule)
    a_elo      = float(row_a.get("elo", 1500.0) or 1500.0)
    b_elo      = float(row_b.get("elo", 1500.0) or 1500.0)
    a_s_elo    = float(row_a.get("strike_elo", 1500.0) or 1500.0)
    b_s_elo    = float(row_b.get("strike_elo", 1500.0) or 1500.0)
    a_g_elo    = float(row_a.get("grapple_elo", 1500.0) or 1500.0)
    b_g_elo    = float(row_b.get("grapple_elo", 1500.0) or 1500.0)
    a_opp_elo  = float(row_a.get("opp_elo_avg", 1500.0) or 1500.0)
    b_opp_elo  = float(row_b.get("opp_elo_avg", 1500.0) or 1500.0)
    a_beat_elo = float(row_a.get("beaten_elo_avg", 1500.0) or 1500.0)
    b_beat_elo = float(row_b.get("beaten_elo_avg", 1500.0) or 1500.0)
    a_mom      = float(row_a.get("elo_momentum", 0.0) or 0.0)
    b_mom      = float(row_b.get("elo_momentum", 0.0) or 0.0)

    elo_expected_a = 1.0 / (1.0 + 10.0 ** ((b_elo - a_elo) / 400.0))
    elo_a_vec = [a_elo, a_s_elo, a_g_elo, a_opp_elo, a_beat_elo, a_mom]
    elo_b_vec = [b_elo, b_s_elo, b_g_elo, b_opp_elo, b_beat_elo, b_mom]
    feats.extend(elo_a_vec)
    feats.extend(elo_b_vec)
    feats.extend([a - b for a, b in zip(elo_a_vec, elo_b_vec)])
    feats.append(elo_expected_a - 0.5)

    # 4. Stance one-hot × 2
    for stats in (row_a, row_b):
        idx = STANCE_MAP.get(str(stats.get("stance", "Orthodox")), 0)
        ohe = [0.0, 0.0, 0.0]
        ohe[idx] = 1.0
        feats.extend(ohe)

    # 5. Matchup de stance
    sa = STANCE_MAP.get(str(row_a.get("stance", "Orthodox")), 0)
    sb = STANCE_MAP.get(str(row_b.get("stance", "Orthodox")), 0)
    feats.append(float(sa != sb))
    feats.append(float(sa == 1))   # A é southpaw
    feats.append(float(sb == 1))   # B é southpaw

    return np.array(feats, dtype=np.float32)


def df_row_to_dicts(row: pd.Series):
    """Converte uma linha do DataFrame em dois dicts (stats_a, stats_b)."""
    stats_a = {c[2:]: row[c] for c in row.index if c.startswith("a_")}
    stats_b = {c[2:]: row[c] for c in row.index if c.startswith("b_")}
    return stats_a, stats_b


def build_Xy(df: pd.DataFrame, col_medians=None):
    """
    Constrói X e y com augmentação por espelhamento.
    col_medians: se None, calcula a partir dos dados (só para treino).
                 Caso contrário, usa os valores passados (val/teste).
    """
    X_orig, X_mirr, y_orig, y_mirr = [], [], [], []

    for _, row in df.iterrows():
        label  = int(row["label"])
        sa, sb = df_row_to_dicts(row)

        # Extrai os streaks gerados pela compute_streaks do Dataframe
        aws = float(row.get("a_win_streak", 0))
        als = float(row.get("a_loss_streak", 0))
        bws = float(row.get("b_win_streak", 0))
        bls = float(row.get("b_loss_streak", 0))

        # Envia os streaks na ordem normal
        X_orig.append(build_row_features(sa, sb, aws, als, bws, bls))
        y_orig.append(label)

        # Na augmentação por espelhamento, A e B trocam de lugar (os streaks também)
        X_mirr.append(build_row_features(sb, sa, bws, bls, aws, als))
        y_mirr.append(1 - label)

    X = np.vstack(X_orig + X_mirr)
    y = np.array(y_orig + y_mirr, dtype=np.int64)

    # Substitui NaN/Inf
    X = np.where(np.isfinite(X), X, np.nan)

    if col_medians is None:
        # Calcula mediana SOMENTE nos dados de treino
        col_medians = np.nanmedian(X, axis=0)

    nan_mask = np.isnan(X)
    X[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])

    return X, y, col_medians


# ══════════════════════════════════════════════════════════════════════════════
# 4. Arquitetura MLP
# ══════════════════════════════════════════════════════════════════════════════

class FightMLP(nn.Module):
    def __init__(self, n_in: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.40),
            nn.Linear(512, 256),  nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.35),
            nn.Linear(256, 128),  nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.25),
            nn.Linear(128, 64),   nn.BatchNorm1d(64),  nn.GELU(), nn.Dropout(0.15),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.net(x)


class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        return logits / self.temperature


# ══════════════════════════════════════════════════════════════════════════════
# 5. Treino / avaliação
# ══════════════════════════════════════════════════════════════════════════════

def make_loader(X, y, shuffle=False, bs=64):
    ds = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, drop_last=shuffle)


def train_epoch(model, loader, opt, crit, device):
    model.train()
    total = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        loss = crit(model(xb), yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total += loss.item() * len(xb)
    return total / len(loader.dataset)


@torch.no_grad()
def eval_model(model, loader, crit, device, ts=None):
    model.eval()
    total, preds, trues, probs = 0, [], [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        if ts is not None:
            logits = ts(logits)
        total += crit(logits, yb).item() * len(xb)
        p = torch.softmax(logits, dim=1)[:, 1]
        preds.extend(logits.argmax(1).cpu().numpy())
        trues.extend(yb.cpu().numpy())
        probs.extend(p.cpu().numpy())
    return total / len(loader.dataset), np.array(preds), np.array(trues), np.array(probs)


def calibrate_temperature(model, val_loader, device, class_weights=None):
    ts = TemperatureScaler().to(device)
    opt = torch.optim.LBFGS([ts.temperature], lr=0.01, max_iter=100)
    
    # Adiciona suporte a pesos para corrigir as probabilidades do método
    if class_weights is not None:
        crit = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        crit = nn.CrossEntropyLoss()

    all_logits, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for xb, yb in val_loader:
            all_logits.append(model(xb.to(device)).cpu())
            all_labels.append(yb)
    logits = torch.cat(all_logits).to(device)
    labels = torch.cat(all_labels).to(device)

    def closure():
        opt.zero_grad()
        loss = crit(ts(logits), labels)
        loss.backward()
        return loss

    opt.step(closure)
    print(f"  Temperatura calibrada: {ts.temperature.item():.4f}")
    return ts


def save_plots(history: dict, te_trues, te_preds, out_dir: str):
    # ── Gráfico de overfitting ────────────────────────────────────────────────
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Curvas de Treinamento", fontsize=14, fontweight="bold")

    # Loss
    axes[0].plot(epochs, history["train_loss"], label="Treino",     color="royalblue")
    axes[0].plot(epochs, history["val_loss"],   label="Validação",  color="tomato")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("CrossEntropy Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Acurácia de validação
    axes[1].plot(epochs, history["val_acc"], color="seagreen", label="Val Acc")
    axes[1].axhline(y=max(history["val_acc"]), color="gray", linestyle="--",
                    alpha=0.6, label=f"Máx: {max(history['val_acc']):.1%}")
    axes[1].set_title("Acurácia — Validação")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("Acurácia")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{out_dir}/training_curves.png", dpi=150)
    plt.close()
    print(f"  📈 Curvas salvas em: {out_dir}/training_curves.png")

    # ── Matriz de confusão ────────────────────────────────────────────────────
    cm   = confusion_matrix(te_trues, te_preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=["B vence", "A vence"])

    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("Matriz de Confusão — Conjunto de Teste", fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/confusion_matrix.png", dpi=150)
    plt.close()
    print(f"  🔢 Matriz salva em:  {out_dir}/confusion_matrix.png")


# ══════════════════════════════════════════════════════════════════════════════
# 6. Pipeline principal
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Dados e split ────────────────────────────────────────────────────────
    df = load_and_clean(DATA_PATH)
    df = compute_streaks(df)          # ← FUNCIONALIDADE NOVA
    df_train, df_val, df_test = temporal_split(df)

    # ── Features ─────────────────────────────────────────────────────────────
    # Mediana calculada SOMENTE no treino
    X_tr, y_tr, col_medians = build_Xy(df_train, col_medians=None)
    X_val, y_val, _         = build_Xy(df_val,   col_medians=col_medians)
    X_te,  y_te,  _         = build_Xy(df_test,  col_medians=col_medians)

    print(f"\nFeatures por amostra: {X_tr.shape[1]}")
    print(f"Amostras treino (c/ augment): {len(X_tr)}")
    print(f"Amostras val    (c/ augment): {len(X_val)}")
    print(f"Amostras teste  (c/ augment): {len(X_te)}")

    # Verificação: nenhum NaN remanescente
    assert not np.isnan(X_tr).any(),  "ERRO: NaN no treino!"
    assert not np.isnan(X_val).any(), "ERRO: NaN na validação!"
    assert not np.isnan(X_te).any(),  "ERRO: NaN no teste!"
    print("  ✅ Zero NaNs em todos os splits")

    # ── Normalização (fit apenas no treino) ──────────────────────────────────
    scaler = StandardScaler()
    X_tr  = scaler.fit_transform(X_tr)
    X_val = scaler.transform(X_val)
    X_te  = scaler.transform(X_te)

    # ── Loaders ──────────────────────────────────────────────────────────────
    tr_loader  = make_loader(X_tr,  y_tr,  shuffle=True,  bs=64)
    val_loader = make_loader(X_val, y_val, shuffle=False, bs=256)
    te_loader  = make_loader(X_te,  y_te,  shuffle=False, bs=256)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # ── Modelo ───────────────────────────────────────────────────────────────
    model     = FightMLP(X_tr.shape[1]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=2e-3)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=5e-4, epochs=200,
        steps_per_epoch=len(tr_loader),
    )

    # ── Loop ─────────────────────────────────────────────────────────────────
    EPOCHS    = 200
    PATIENCE  = 40
    best_val  = float("inf")
    no_improv = 0
    history   = {"train_loss": [], "val_loss": [], "val_acc": []}

    print(f"\n{'Ep':>4}  {'Train':>8}  {'Val':>8}  {'Val Acc':>8}")
    print("─" * 38)

    for ep in range(1, EPOCHS + 1):
        tr_l = train_epoch(model, tr_loader, optimizer, criterion, device)
        scheduler.step()
        val_l, preds, trues, _ = eval_model(model, val_loader, criterion, device)
        acc = accuracy_score(trues, preds)

        history["train_loss"].append(float(tr_l))
        history["val_loss"].append(float(val_l))
        history["val_acc"].append(float(acc))

        if ep % 20 == 0 or ep == 1:
            print(f"{ep:>4}  {tr_l:>8.4f}  {val_l:>8.4f}  {acc:>7.1%}")

        if val_l < best_val - 1e-5:
            best_val  = val_l
            no_improv = 0
            torch.save(model.state_dict(), f"{OUT_DIR}/best_model.pt")
        else:
            no_improv += 1
            if no_improv >= PATIENCE:
                print(f"\n  Early stopping na época {ep} (melhor val_loss: {best_val:.4f})")
                break

    # ── Calibração de temperatura (usando apenas validação) ──────────────────
    model.load_state_dict(torch.load(f"{OUT_DIR}/best_model.pt", map_location=device))
    print("\nCalibrando temperatura com conjunto de validação...")
    ts = calibrate_temperature(model, val_loader, device)
    torch.save(ts.state_dict(), f"{OUT_DIR}/temperature.pt")

    # ── Avaliação final no TESTE (primeira e única vez que o teste é usado) ──
    _, te_preds, te_trues, te_probs = eval_model(
        model, te_loader, criterion, device, ts=ts)

    acc = accuracy_score(te_trues, te_preds)
    auc = roc_auc_score(te_trues, te_probs)
    ll  = log_loss(te_trues, te_probs)
    bs  = brier_score_loss(te_trues, te_probs)

    print("\n" + "═" * 62)
    print("  📊 AVALIAÇÃO FINAL — Conjunto de TESTE (nunca visto antes)")
    print("═" * 62)
    print(f"  Lutas no teste  : {len(df_test)} ({len(X_te)} c/ augmentação)")
    print(f"  Período         : {df_test['event_date'].min().date()} → {df_test['event_date'].max().date()}")
    print(f"  Acurácia        : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"  ROC-AUC         : {auc:.4f}")
    print(f"  Log-Loss        : {ll:.4f}")
    print(f"  Brier Score     : {bs:.4f}  (0=perfeito, 0.25=aleatório)")
    print()
    print(classification_report(te_trues, te_preds,
                                 target_names=["B vence", "A vence"]))

    print("  Confiança vs Acurácia:")
    for thr in [0.50, 0.55, 0.60, 0.65, 0.70]:
        mask = (te_probs >= thr) | (te_probs <= 1 - thr)
        n_m  = mask.sum()
        if n_m == 0:
            print(f"    |P − 0.5| ≥ {thr-0.5:.2f}  →     0 amostras  (modelo nunca atinge essa confiança)")
            break  

        correct = ((te_probs[mask] >= 0.5) == te_trues[mask].astype(bool)).sum()
        acc_t   = correct / n_m

        ci_lo = binom.interval(0.95, n_m, acc_t)[0] / n_m
        ci_hi = binom.interval(0.95, n_m, acc_t)[1] / n_m

        flag = "  ⚠️  (n pequeno)" if n_m < 50 else ""
        print(f"    |P − 0.5| ≥ {thr-0.5:.2f}  →  {n_m:>4} amostras  "
            f"acc={acc_t:.1%}  [IC95%: {ci_lo:.1%}–{ci_hi:.1%}]{flag}")

    # ── Salvar artefatos ─────────────────────────────────────────────────────
    with open(f"{OUT_DIR}/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    np.save(f"{OUT_DIR}/col_medians.npy", col_medians)

    meta = {
        "n_features":   int(X_tr.shape[1]),
        "numeric_base": NUMERIC_BASE,
        "stance_map":   STANCE_MAP,
        "history":      history,
        "test_acc":     float(acc),
        "test_auc":     float(auc),
        "test_logloss": float(ll),
        "test_brier":   float(bs),
        "train_period": [str(df_train["event_date"].min().date()),
                         str(df_train["event_date"].max().date())],
        "val_period":   [str(df_val["event_date"].min().date()),
                         str(df_val["event_date"].max().date())],
        "test_period":  [str(df_test["event_date"].min().date()),
                         str(df_test["event_date"].max().date())],
    }
    with open(f"{OUT_DIR}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✅ Artefatos salvos em: {OUT_DIR}")

    print("\nGerando gráficos...")
    save_plots(history, te_trues, te_preds, OUT_DIR)


if __name__ == "__main__":
    main()