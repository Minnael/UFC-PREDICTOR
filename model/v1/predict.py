"""
UFC Fight Predictor — Inferência
=================================
Uso:
    from predict import FightPredictor
    p = FightPredictor("artifacts/")
    result = p.predict(fighter_a_stats, fighter_b_stats)
    print(result)
"""

import json, pickle, os
import numpy as np
import torch
import torch.nn as nn
from typing import Optional


STANCE_MAP   = {"Orthodox": 0, "Southpaw": 1, "Switch": 2}

NUMERIC_BASE = [
    "height_cm", "weight_kg", "reach_arm_cm", "age",
    "wins", "losses", "draws", "wins_ko", "wins_sub", "wins_dec",
    "total_fights",
    "sig_strikes_hit_total", "td_hit_total", "sub_att_total", "kd_total",
    "sig_strikes_per_min", "td_per_15min", "sub_per_15min",
    "avg_fight_time_sec",
]


# ══════════════════════════════════════════════════════════════════════════════
# Arquitetura (deve ser idêntica ao train_v2.py)
# ══════════════════════════════════════════════════════════════════════════════

class FightMLP(nn.Module):
    def __init__(self, n_in, hidden=(512, 256, 128, 64), dropout=(0.4, 0.35, 0.25, 0.15)):
        super().__init__()
        layers = []
        prev = n_in
        for h, d in zip(hidden, dropout):
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.GELU(),
                nn.Dropout(d),
            ]
            prev = h
        layers.append(nn.Linear(prev, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        return logits / self.temperature


# ══════════════════════════════════════════════════════════════════════════════
# Feature engineering (espelho de train_v2.py)
# ══════════════════════════════════════════════════════════════════════════════

def _derived(stats: dict) -> list:
    wins   = float(stats.get("wins",   0) or 0)
    losses = float(stats.get("losses", 0) or 0)
    draws  = float(stats.get("draws",  0) or 0)
    total  = float(stats.get("total_fights", 0) or 0)
    ko     = float(stats.get("wins_ko",  0) or 0)
    sub    = float(stats.get("wins_sub", 0) or 0)
    dec    = float(stats.get("wins_dec", 0) or 0)

    win_rate    = wins / total   if total > 0 else 0.5
    finish_rate = (ko + sub) / wins if wins > 0 else 0.0
    ko_rate     = ko  / wins if wins > 0 else 0.0
    sub_rate    = sub / wins if wins > 0 else 0.0
    dec_rate    = dec / wins if wins > 0 else 0.0
    loss_rate   = losses / total if total > 0 else 0.5
    exp_score   = np.log1p(total)
    td_15  = float(stats.get("td_per_15min",  0) or 0)
    sub_15 = float(stats.get("sub_per_15min", 0) or 0)
    grapple_score = td_15 + sub_15 * 0.5

    return [
        win_rate, finish_rate, ko_rate, sub_rate, dec_rate, loss_rate,
        exp_score, grapple_score,
    ]


def _build_features(stats_a: dict, stats_b: dict) -> np.ndarray:
    feats = []

    # 1. Deltas (A − B)
    for col in NUMERIC_BASE:
        va = float(stats_a.get(col, 0) or 0)
        vb = float(stats_b.get(col, 0) or 0)
        feats.append(va - vb)

    # 2. Valores absolutos A, depois B
    for stats in (stats_a, stats_b):
        for col in NUMERIC_BASE:
            feats.append(float(stats.get(col, 0) or 0))

    # 3. Features derivadas + deltas
    da = _derived(stats_a)
    db = _derived(stats_b)
    feats.extend(da)
    feats.extend(db)
    feats.extend([a - b for a, b in zip(da, db)])

    # 4. Stance one-hot × 2
    for stats in (stats_a, stats_b):
        stance = str(stats.get("stance", "Orthodox"))
        idx    = STANCE_MAP.get(stance, 0)
        ohe    = [0.0, 0.0, 0.0]
        ohe[idx] = 1.0
        feats.extend(ohe)

    # 5. Stance matchup
    sa = STANCE_MAP.get(str(stats_a.get("stance", "Orthodox")), 0)
    sb = STANCE_MAP.get(str(stats_b.get("stance", "Orthodox")), 0)
    feats.append(float(sa != sb))
    feats.append(float(sa == 1))
    feats.append(float(sb == 1))

    return np.array(feats, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Predictor
# ══════════════════════════════════════════════════════════════════════════════

class FightPredictor:
    """
    Carrega os artefatos e realiza predições.

    Exemplo de stats dict:
        {
            "height_cm": 180.3,
            "weight_kg": 77.1,
            "reach_arm_cm": 185.4,
            "age": 29.6,
            "wins": 9,
            "losses": 1,
            "draws": 0,
            "wins_ko": 4,
            "wins_sub": 1,
            "wins_dec": 4,
            "total_fights": 10,
            "sig_strikes_hit_total": 667,
            "td_hit_total": 1,
            "sub_att_total": 1,
            "kd_total": 6,
            "sig_strikes_per_min": 5.565,
            "td_per_15min": 0.125,
            "sub_per_15min": 0.125,
            "avg_fight_time_sec": 719.1,
            "stance": "Switch",
        }
    """

    def __init__(self, artifacts_dir: str = "artifacts"):
        self.device = torch.device("cpu")

        with open(f"{artifacts_dir}/meta.json") as f:
            meta = json.load(f)

        self.col_medians = np.load(f"{artifacts_dir}/col_medians.npy")

        with open(f"{artifacts_dir}/scaler.pkl", "rb") as f:
            self.scaler = pickle.load(f)

        self.model = FightMLP(meta["n_features"]).to(self.device)
        self.model.load_state_dict(
            torch.load(f"{artifacts_dir}/best_model.pt", map_location=self.device)
        )
        self.model.eval()

        self.ts = TemperatureScaler().to(self.device)
        self.ts.load_state_dict(
            torch.load(f"{artifacts_dir}/temperature.pt", map_location=self.device)
        )

    def predict(
        self,
        fighter_a: dict,
        fighter_b: dict,
        name_a: str = "Fighter A",
        name_b: str = "Fighter B",
    ) -> dict:
        """
        Retorna dicionário com probabilidades e diagnóstico.
        """
        x = _build_features(fighter_a, fighter_b)

        # Imputa NaN/Inf pela mediana de treino
        x = np.where(np.isfinite(x), x, np.nan)
        nan_idx = np.where(np.isnan(x))[0]
        for i in nan_idx:
            x[i] = self.col_medians[i] if i < len(self.col_medians) else 0.0

        x_scaled = self.scaler.transform(x.reshape(1, -1))
        tensor   = torch.FloatTensor(x_scaled).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            logits = self.ts(logits)
            probs  = torch.softmax(logits, dim=1)[0]

        prob_a = float(probs[1])
        prob_b = float(probs[0])

        # Confiança: quão longe de 50/50
        confidence = abs(prob_a - 0.5) * 2   # 0 = incerto, 1 = certeza total

        if confidence < 0.10:
            conf_label = "⚪ Incerta (toss-up)"
        elif confidence < 0.20:
            conf_label = "🟡 Moderada"
        elif confidence < 0.35:
            conf_label = "🟠 Alta"
        else:
            conf_label = "🔴 Muito Alta"

        favorite = name_a if prob_a > prob_b else name_b

        return {
            "fighter_a":  name_a,
            "fighter_b":  name_b,
            "prob_a":     round(prob_a * 100, 1),
            "prob_b":     round(prob_b * 100, 1),
            "favorite":   favorite,
            "confidence": round(confidence * 100, 1),
            "conf_label": conf_label,
        }

    def print_prediction(self, result: dict):
        pa = result["prob_a"]
        pb = result["prob_b"]
        na = result["fighter_a"]
        nb = result["fighter_b"]
        bar_a = "█" * int(pa / 2)
        bar_b = "█" * int(pb / 2)
        print("\n" + "═" * 55)
        print("  🥊  UFC FIGHT PREDICTOR")
        print("═" * 55)
        print(f"  {na:<22} vs  {nb}")
        print("─" * 55)
        print(f"  {na:<22} {pa:>5.1f}%  {bar_a}")
        print(f"  {nb:<22} {pb:>5.1f}%  {bar_b}")
        print("─" * 55)
        print(f"  Favorito    : {result['favorite']}")
        print(f"  Confiança   : {result['conf_label']}  ({result['confidence']:.1f}%)")
        print("═" * 55)


# ══════════════════════════════════════════════════════════════════════════════
# Exemplo de uso direto
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Importa dados para buscar as stats dos lutadores de exemplo
    import pandas as pd

    df = pd.read_csv("/mnt/user-data/uploads/ufc_dataset.csv")

    # Pega a primeira luta como referência (Della Maddalena vs Prates)
    row = df.iloc[0]

    stats_della = {col[2:]: row[col] for col in df.columns if col.startswith("a_")}
    stats_prates = {col[2:]: row[col] for col in df.columns if col.startswith("b_")}

    predictor = FightPredictor("artifacts")
    result = predictor.predict(
        stats_della,
        stats_prates,
        name_a="Jack Della Maddalena",
        name_b="Carlos Prates",
    )
    predictor.print_prediction(result)

    print(f"\n  (Resultado real: {row['winner']} venceu — {'Della Maddalena' if row['winner']=='A' else 'Carlos Prates'})")
