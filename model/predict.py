"""
UFC Fight Predictor — predict.py
=================================
Carrega os artefatos Multi-Task (V5) e realiza predições completas para novas lutas.
"""

import os, pickle
import numpy as np
import torch
import torch.nn as nn

STANCE_MAP   = {"Orthodox": 0, "Southpaw": 1, "Switch": 2}
NUMERIC_BASE = [
    "height_cm", "weight_kg", "reach_arm_cm", "age",
    "wins", "losses", "draws", "wins_ko", "wins_sub", "wins_dec",
    "total_fights",
    "sig_strikes_hit_total", "td_hit_total", "sub_att_total", "kd_total",
    "sig_strikes_per_min", "td_per_15min", "sub_per_15min",
    "avg_fight_time_sec",
    "last_fight_1", "last_fight_2", "last_fight_3", "last_fight_4", "last_fight_5",
]
# V5 usa as streaks
NUMERIC_BASE_V5 = NUMERIC_BASE + ["win_streak", "loss_streak"]

class MultiTaskFightMLP(nn.Module):
    def __init__(self, n_in: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(n_in, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.40),
            nn.Linear(512, 256),  nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.35),
            nn.Linear(256, 128),  nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.25),
        )
        self.head_winner = nn.Sequential(
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(64, 2) # V5 usa 2 classes na saída
        )
        self.head_method = nn.Sequential(
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        features = self.shared(x)
        return self.head_winner(features), self.head_method(features)

class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        return logits / self.temperature

class FightPredictor:
    def __init__(self, artifacts_dir: str):
        with open(f"{artifacts_dir}/scaler.pkl", "rb") as f:
            self.scaler = pickle.load(f)
        self.col_medians = np.load(f"{artifacts_dir}/col_medians.npy")
        
        n_features = len(self.col_medians)
        self.model = MultiTaskFightMLP(n_features)
        
        # Carrega os pesos do V5 (agora em .pt)
        model_path = f"{artifacts_dir}/best_model.pt"
        self.model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        self.model.eval()

        # Carrega a Temperatura para calibrar as percentagens
        self.ts = TemperatureScaler()
        ts_path = f"{artifacts_dir}/temperature.pt"
        if os.path.exists(ts_path):
            self.ts.load_state_dict(torch.load(ts_path, map_location="cpu", weights_only=True))
        self.ts.eval()

    def _derived(self, stats: dict) -> list:
        wins   = float(stats.get("wins",   0) or 0)
        total  = float(stats.get("total_fights", 0) or 0)
        ko     = float(stats.get("wins_ko",  0) or 0)
        sub    = float(stats.get("wins_sub", 0) or 0)
        dec    = float(stats.get("wins_dec", 0) or 0)
        losses = float(stats.get("losses",   0) or 0)
        td_15  = float(stats.get("td_per_15min",  0) or 0)
        sub_15 = float(stats.get("sub_per_15min", 0) or 0)

        return [
            wins / total        if total > 0 else 0.5,
            (ko + sub) / wins   if wins  > 0 else 0.0,
            ko  / wins          if wins  > 0 else 0.0,
            sub / wins          if wins  > 0 else 0.0,
            dec / wins          if wins  > 0 else 0.0,
            losses / total      if total > 0 else 0.5,
            float(np.log1p(total)),
            td_15 + sub_15 * 0.5,
        ]

    def _prepare_vector(self, row_a: dict, row_b: dict) -> np.ndarray:
        # Garante que streaks faltantes sejam 0 na inferência ao vivo
        for k in ["win_streak", "loss_streak"]:
            if k not in row_a: row_a[k] = 0
            if k not in row_b: row_b[k] = 0

        feats = []
        for col in NUMERIC_BASE_V5:
            va = float(row_a.get(col, 0) or 0)
            vb = float(row_b.get(col, 0) or 0)
            feats.append(va - vb)
        for stats in (row_a, row_b):
            for col in NUMERIC_BASE_V5:
                feats.append(float(stats.get(col, 0) or 0))
        da = self._derived(row_a)
        db = self._derived(row_b)
        feats.extend(da)
        feats.extend(db)
        feats.extend([a - b for a, b in zip(da, db)])
        for stats in (row_a, row_b):
            idx = STANCE_MAP.get(str(stats.get("stance", "Orthodox")), 0)
            ohe = [0.0, 0.0, 0.0]
            ohe[idx] = 1.0
            feats.extend(ohe)
        sa = STANCE_MAP.get(str(row_a.get("stance", "Orthodox")), 0)
        sb = STANCE_MAP.get(str(row_b.get("stance", "Orthodox")), 0)
        feats.append(float(sa != sb))
        feats.append(float(sa == 1))
        feats.append(float(sb == 1))
        
        vec = np.array(feats, dtype=np.float32).reshape(1, -1)
        vec[np.isnan(vec)] = np.take(self.col_medians, np.where(np.isnan(vec))[1])
        return self.scaler.transform(vec)

    def predict(self, fighter_a: dict, fighter_b: dict, name_a="Fighter A", name_b="Fighter B") -> dict:
        vec_ab = self._prepare_vector(fighter_a, fighter_b)
        vec_ba = self._prepare_vector(fighter_b, fighter_a)

        tensor_ab = torch.FloatTensor(vec_ab)
        tensor_ba = torch.FloatTensor(vec_ba)

        with torch.no_grad():
            win_logits_ab, met_logits_ab = self.model(tensor_ab)
            win_logits_ba, met_logits_ba = self.model(tensor_ba)

            # 1. Calibrar logits do vencedor com a Temperatura
            win_logits_ab = self.ts(win_logits_ab)
            win_logits_ba = self.ts(win_logits_ba)

            # 2. Calcular probabilidades do Vencedor (Média Simétrica)
            # O índice 1 significa que o primeiro lutador passado (A) vence
            prob_a_ab = torch.softmax(win_logits_ab, dim=1)[0, 1].item()
            
            # Na perspetiva invertida, o índice 1 significa que o B vence, logo A é o índice 0
            prob_a_ba = torch.softmax(win_logits_ba, dim=1)[0, 0].item() 
            
            prob_a = (prob_a_ab + prob_a_ba) / 2.0
            prob_b = 1.0 - prob_a

            # 3. Calcular probabilidades do Método (Soma Simétrica)
            met_logits_final = (met_logits_ab + met_logits_ba) / 2.0
            met_probs = torch.softmax(met_logits_final, dim=1)[0].numpy()
            
            prob_dec = float(met_probs[0])
            prob_ko  = float(met_probs[1])
            prob_sub = float(met_probs[2])

        confidence = abs(prob_a - 0.5) * 2

        if confidence < 0.10:   conf_label = "⚪ Incerta (toss-up)"
        elif confidence < 0.20: conf_label = "🟡 Moderada"
        elif confidence < 0.30: conf_label = "🟠 Alta"
        else:                   conf_label = "🔴 Muito Alta"

        return {
            "fighter_a":  name_a,
            "fighter_b":  name_b,
            "prob_a":     round(prob_a * 100, 1),
            "prob_b":     round(prob_b * 100, 1),
            "favorite":   name_a if prob_a > prob_b else name_b,
            "confidence": round(confidence * 100, 1),
            "conf_label": conf_label,
            "method_probs": {
                "Decisão": round(prob_dec * 100, 1),
                "Nocaute": round(prob_ko * 100, 1),
                "Finalização": round(prob_sub * 100, 1)
            }
        }