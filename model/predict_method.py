"""
UFC Fight Method Predictor — predict_method.py
================================================
Combina o modelo de vitória JÁ EXISTENTE (predict.py / artifacts/) com o
modelo de método (train_method.py / artifacts_method/) para produzir a
probabilidade de cada desfecho: (vencedor × método).

Não modifica predict.py — importa e reaproveita FightPredictor e
_build_features de lá, para garantir que a engenharia de features usada
aqui seja EXATAMENTE a mesma usada no modelo de vitória.
"""

import json, pickle
import numpy as np
import torch
import torch.nn as nn

from model.predict import FightPredictor, _build_features

METHOD_NAMES = ["KO/TKO", "Submissão", "Decisão"]


# ── Arquitetura (idêntica à definida em train_method.py) ─────────────────────

class MethodMLP(nn.Module):
    def __init__(self, n_in: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.40),
            nn.Linear(512, 256),  nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.35),
            nn.Linear(256, 128),  nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.25),
            nn.Linear(128, 64),   nn.BatchNorm1d(64),  nn.GELU(), nn.Dropout(0.15),
            nn.Linear(64, 3),
        )

    def forward(self, x):
        return self.net(x)


class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        return logits / self.temperature


class MethodPredictor:
    def __init__(self, artifacts_dir: str = "artifacts_method"):
        self.device = torch.device("cpu")

        with open(f"{artifacts_dir}/meta.json") as f:
            meta = json.load(f)

        self.col_medians = np.load(f"{artifacts_dir}/col_medians.npy")

        with open(f"{artifacts_dir}/scaler.pkl", "rb") as f:
            self.scaler = pickle.load(f)

        self.model = MethodMLP(meta["n_features"]).to(self.device)
        self.model.load_state_dict(
            torch.load(f"{artifacts_dir}/best_model.pt", map_location=self.device)
        )
        self.model.eval()

        self.ts = TemperatureScaler().to(self.device)
        self.ts.load_state_dict(
            torch.load(f"{artifacts_dir}/temperature.pt", map_location=self.device)
        )

    def predict_conditional(self, winner_stats: dict, loser_stats: dict) -> np.ndarray:
        """P(método | winner_stats vence loser_stats). Retorna array shape (3,)
        na ordem METHOD_NAMES = ['KO/TKO', 'Submissão', 'Decisão']."""
        x = _build_features(winner_stats, loser_stats)
        x = np.where(np.isfinite(x), x, np.nan)
        nan_idx = np.where(np.isnan(x))[0]
        for i in nan_idx:
            x[i] = self.col_medians[i] if i < len(self.col_medians) else 0.0

        x_scaled = self.scaler.transform(x.reshape(1, -1))
        tensor   = torch.FloatTensor(x_scaled).to(self.device)

        with torch.no_grad():
            logits = self.ts(self.model(tensor))
            probs  = torch.softmax(logits, dim=1)[0]

        return probs.numpy()


class FullPredictor:
    """Combina vitória + método via probabilidade total:
    P(A vence por KO) = P(A vence) x P(método=KO | A vence)."""

    def __init__(self, win_artifacts: str = "artifacts",
                 method_artifacts: str = "artifacts_method"):
        self.win_model    = FightPredictor(win_artifacts)
        self.method_model = MethodPredictor(method_artifacts)

    def predict(self, fighter_a: dict, fighter_b: dict,
                name_a: str = "Fighter A", name_b: str = "Fighter B") -> dict:
        win_result = self.win_model.predict(fighter_a, fighter_b, name_a, name_b)
        p_a = win_result["prob_a"] / 100
        p_b = win_result["prob_b"] / 100

        # Consulta o modelo de método assumindo cada lado como vencedor
        p_method_if_a = self.method_model.predict_conditional(fighter_a, fighter_b)
        p_method_if_b = self.method_model.predict_conditional(fighter_b, fighter_a)

        outcomes = {}
        for i, metodo in enumerate(METHOD_NAMES):
            outcomes[f"{name_a} — {metodo}"] = round(float(p_a * p_method_if_a[i]) * 100, 1)
        for i, metodo in enumerate(METHOD_NAMES):
            outcomes[f"{name_b} — {metodo}"] = round(float(p_b * p_method_if_b[i]) * 100, 1)

        return {
            "fighter_a": name_a,
            "fighter_b": name_b,
            "prob_a":    win_result["prob_a"],
            "prob_b":    win_result["prob_b"],
            "favorite":  win_result["favorite"],
            "confidence": win_result["confidence"],
            "conf_label": win_result["conf_label"],
            "outcomes":  outcomes,   # 6 probabilidades — soma = 100%
        }

    def print_prediction(self, result: dict):
        print("\n" + "═" * 60)
        print(f"  {result['fighter_a']} vs {result['fighter_b']}")
        print("─" * 60)
        for desfecho, pct in sorted(result["outcomes"].items(), key=lambda kv: -kv[1]):
            bar = "█" * int(pct / 2)
            print(f"  {desfecho:<32} {pct:>5.1f}%  {bar}")
        print("─" * 60)
        print(f"  Favorito: {result['favorite']}")
        print("═" * 60)
