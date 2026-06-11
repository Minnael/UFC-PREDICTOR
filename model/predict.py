"""
UFC Fight Predictor — predict.py
=================================
Carrega os artefatos Multi-Task e realiza predições completas para novas lutas.
"""

import pickle
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

class MultiTaskFightMLP(nn.Module):
    def __init__(self, n_in: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(n_in, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.40),
            nn.Linear(512, 256),  nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.30),
            nn.Linear(256, 128),  nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.20),
        )
        self.head_winner = nn.Sequential(
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(64, 1)
        )
        self.head_method = nn.Sequential(
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        features = self.shared(x)
        win_logits = self.head_winner(features)
        met_logits = self.head_method(features)
        return win_logits, met_logits

class FightPredictor:
    def __init__(self, artifacts_dir: str):
        with open(f"{artifacts_dir}/scaler.pkl", "rb") as f:
            self.scaler = pickle.load(f)
        self.col_medians = np.load(f"{artifacts_dir}/col_medians.npy")
        
        n_features = len(self.col_medians)
        self.model = MultiTaskFightMLP(n_features)
        
        model_path = f"{artifacts_dir}/best_model.pth"
        self.model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        self.model.eval()

    def _prepare_vector(self, fa: dict, fb: dict) -> np.ndarray:
        vec_a = [fa.get(c, -1.0) for c in NUMERIC_BASE]
        vec_b = [fb.get(c, -1.0) for c in NUMERIC_BASE]
        
        st_a = STANCE_MAP.get(fa.get("stance", ""), -1)
        st_b = STANCE_MAP.get(fb.get("stance", ""), -1)
        
        vec = np.array(vec_a + [st_a] + vec_b + [st_b], dtype=np.float32).reshape(1, -1)
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

            # 1. Resolver o Vencedor (Subtração Simétrica)
            win_logit_final = (win_logits_ab - win_logits_ba) / 2.0
            prob_a = torch.sigmoid(win_logit_final).item()
            prob_b = 1.0 - prob_a

            # 2. Resolver o Método (Soma Simétrica, pois o método é o mesmo independente da perspectiva)
            met_logits_final = (met_logits_ab + met_logits_ba) / 2.0
            met_probs = torch.softmax(met_logits_final, dim=1)[0].numpy()
            
            # As ordens das classes foram definidas no preparar_metodos.py:
            # 0 = Decisão, 1 = Nocaute, 2 = Finalização
            prob_dec = float(met_probs[0])
            prob_ko  = float(met_probs[1])
            prob_sub = float(met_probs[2])

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
            "method_probs": {
                "Decisão": round(prob_dec * 100, 1),
                "Nocaute": round(prob_ko * 100, 1),
                "Finalização": round(prob_sub * 100, 1)
            }
        }