"""
UFC Fight Predictor — predict.py
=================================
Carrega os artefatos treinados e realiza predições para novas lutas.
Não modifique este arquivo.
"""

import json, pickle
import numpy as np
import torch
import torch.nn as nn

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
    "last_fight_1", "last_fight_2", "last_fight_3", "last_fight_4", "last_fight_5",
]


# ── Arquitetura (idêntica ao train.py) ─────────────────────────────────────

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


# ── Feature engineering (idêntico ao train.py) ────────────────────────────

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
        wins / total        if total > 0 else 0.5,
        (ko + sub) / wins   if wins  > 0 else 0.0,
        ko  / wins          if wins  > 0 else 0.0,
        sub / wins          if wins  > 0 else 0.0,
        dec / wins          if wins  > 0 else 0.0,
        losses / total      if total > 0 else 0.5,
        float(np.log1p(total)),
        td_15 + sub_15 * 0.5,
        (l_ko + l_sub) / total  if total > 0 else 0.0,  # been_finished_rate
        l_ko  / total           if total > 0 else 0.0,  # ko_vulnerability
        l_sub / total           if total > 0 else 0.0,  # sub_vulnerability
    ]


def _impute_debutant(stats: dict) -> dict:
    if int(float(stats.get("total_fights", 0) or 0)) > 0:
        return stats
    
    s = stats.copy()
    s["total_fights"] = 3.0
    s["wins"] = 1.5
    s["losses"] = 1.5
    s["wins_ko"] = 0.45
    s["wins_sub"] = 0.30
    s["wins_dec"] = 0.75
    s["losses_ko"] = 0.45
    s["losses_sub"] = 0.30
    s["losses_dec"] = 0.75
    s["sig_strikes_hit_total"] = 105.0
    s["td_hit_total"] = 3.6
    s["sub_att_total"] = 1.2
    s["kd_total"] = 0.3
    s["sig_strikes_per_min"] = 3.5
    s["td_per_15min"] = 1.2
    s["sub_per_15min"] = 0.4
    
    s["kd_absorbed_total"] = 0.3
    s["sig_strikes_absorbed_total"] = 105.0
    s["td_absorbed_total"] = 3.6
    s["sub_absorbed_total"] = 1.2
    s["sig_strikes_absorbed_per_min"] = 3.5
    s["td_absorbed_per_15min"] = 1.2
    s["sub_absorbed_per_15min"] = 0.4
    
    s["avg_fight_time_sec"] = 600.0
    return s


def _build_features(stats_a: dict, stats_b: dict) -> np.ndarray:
    stats_a = _impute_debutant(stats_a)
    stats_b = _impute_debutant(stats_b)
    feats = []
    for col in NUMERIC_BASE:
        va = float(stats_a.get(col, 0) or 0)
        vb = float(stats_b.get(col, 0) or 0)
        feats.append(va - vb)
    for stats in (stats_a, stats_b):
        for col in NUMERIC_BASE:
            feats.append(float(stats.get(col, 0) or 0))
    da = _derived(stats_a)
    db = _derived(stats_b)
    feats.extend(da)
    feats.extend(db)
    feats.extend([a - b for a, b in zip(da, db)])
    for stats in (stats_a, stats_b):
        idx = STANCE_MAP.get(str(stats.get("stance", "Orthodox")), 0)
        ohe = [0.0, 0.0, 0.0]
        ohe[idx] = 1.0
        feats.extend(ohe)
    sa = STANCE_MAP.get(str(stats_a.get("stance", "Orthodox")), 0)
    sb = STANCE_MAP.get(str(stats_b.get("stance", "Orthodox")), 0)
    feats.append(float(sa != sb))
    feats.append(float(sa == 1))
    feats.append(float(sb == 1))
    return np.array(feats, dtype=np.float32)


# ── Predictor ─────────────────────────────────────────────────────────────────

class FightPredictor:
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

    def predict(self, fighter_a: dict, fighter_b: dict,
                name_a: str = "Fighter A", name_b: str = "Fighter B") -> dict:
        x = _build_features(fighter_a, fighter_b)
        x = np.where(np.isfinite(x), x, np.nan)
        nan_idx = np.where(np.isnan(x))[0]
        for i in nan_idx:
            x[i] = self.col_medians[i] if i < len(self.col_medians) else 0.0

        x_scaled = self.scaler.transform(x.reshape(1, -1))
        tensor   = torch.FloatTensor(x_scaled).to(self.device)

        with torch.no_grad():
            logits = self.ts(self.model(tensor))
            probs  = torch.softmax(logits, dim=1)[0]

        prob_a = float(probs[1])
        prob_b = float(probs[0])
        confidence = abs(prob_a - 0.5) * 2

        if confidence < 0.10:   conf_label = "⚪ Incerta (toss-up)"
        elif confidence < 0.20: conf_label = "🟡 Moderada"
        elif confidence < 0.35: conf_label = "🟠 Alta"
        else:                   conf_label = "🔴 Muito Alta"

        return {
            "fighter_a":  name_a,
            "fighter_b":  name_b,
            "prob_a":     round(prob_a * 100, 1),
            "prob_b":     round(prob_b * 100, 1),
            "favorite":   name_a if prob_a > prob_b else name_b,
            "confidence": round(confidence * 100, 1),
            "conf_label": conf_label,
        }

    def print_prediction(self, result: dict):
        pa, pb   = result["prob_a"], result["prob_b"]
        na, nb   = result["fighter_a"], result["fighter_b"]
        bar_a    = "█" * int(pa / 2)
        bar_b    = "█" * int(pb / 2)
        print("\n" + "═" * 58)
        print("  🥊  UFC FIGHT PREDICTOR")
        print("═" * 58)
        print(f"  {na} vs {nb}")
        print("─" * 58)
        print(f"  {na:<26} {pa:>5.1f}%  {bar_a}")
        print(f"  {nb:<26} {pb:>5.1f}%  {bar_b}")
        print("─" * 58)
        print(f"  Favorito  : {result['favorite']}")
        print(f"  Confiança : {result['conf_label']}  ({result['confidence']:.1f}%)")
        print("═" * 58)
