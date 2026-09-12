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


def build_row_features(stats_a: dict, stats_b: dict, a_win_streak: float=0, a_loss_streak: float=0, b_win_streak: float=0, b_loss_streak: float=0) -> np.ndarray:
    # Observe que as duas linhas antigas de imputação SUMIRAM daqui.
    feats = []

    # 1. Deltas numéricos (A − B)
    for col in NUMERIC_BASE:
        va = float(stats_a.get(col, 0) or 0)
        vb = float(stats_b.get(col, 0) or 0)
        feats.append(va - vb)

    # 2. Absolutos
    for stats in (stats_a, stats_b):
        for col in NUMERIC_BASE:
            feats.append(float(stats.get(col, 0) or 0))

    # 3. Derivadas
    da = _derived(stats_a)
    db = _derived(stats_b)
    feats.extend(da)
    feats.extend(db)
    feats.extend([a - b for a, b in zip(da, db)])

    # 4. STREAKS (As novas features)
    feats.extend([a_win_streak, a_loss_streak, b_win_streak, b_loss_streak])
    feats.extend([a_win_streak - b_win_streak, a_loss_streak - b_loss_streak])

    # 5. Stances
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
        
        # TRAVA DE SEGURANÇA: Bloqueia a previsão se algum deles for estreante puro
        if float(fighter_a.get("total_fights", 0)) == 0 or float(fighter_b.get("total_fights", 0)) == 0:
            return {
                "fighter_a": name_a,
                "fighter_b": name_b,
                "error": "Previsão abortada: O modelo exige lutadores com pelo menos 1 luta no histórico do UFC."
            }

        # EXTRAÇÃO DOS STREAKS: Puxa as sequências se existirem (padrão 0)
        aws = float(fighter_a.get("win_streak", 0))
        als = float(fighter_a.get("loss_streak", 0))
        bws = float(fighter_b.get("win_streak", 0))
        bls = float(fighter_b.get("loss_streak", 0))

        # Passa os dicionários puramente reais e os streaks para construir as features
        x = build_row_features(fighter_a, fighter_b, aws, als, bws, bls)
        
        # A partir daqui, o código flui normalmente (Nan check, Scaler, Tensor, etc)
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