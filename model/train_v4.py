"""
UFC Fight Outcome & Method Predictor — train_v4.py
===================================================
Arquitetura Multi-Task:
  - Cabeça 1: Previsão do Vencedor (A ou B)
  - Cabeça 2: Previsão do Método (0=Decisão, 1=Nocaute, 2=Finalização)
"""

import os, json, pickle, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, log_loss, brier_score_loss,
    classification_report
)
from scipy.stats import binom
warnings.filterwarnings("ignore")

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DATA_PATH  = "ufc_dataset_v4.csv" # Base nova com a coluna method_class
OUT_DIR    = "artifacts"
os.makedirs(OUT_DIR, exist_ok=True)

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

# ── Arquitetura Multi-Task ──────────────────────────────────────────
class MultiTaskFightMLP(nn.Module):
    def __init__(self, n_in: int):
        super().__init__()
        # Tronco Compartilhado
        self.shared = nn.Sequential(
            nn.Linear(n_in, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.40),
            nn.Linear(512, 256),  nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.30),
            nn.Linear(256, 128),  nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.20),
        )
        # Cabeça 1: Vencedor (Binário)
        self.head_winner = nn.Sequential(
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(64, 1)
        )
        # Cabeça 2: Método (3 Classes)
        self.head_method = nn.Sequential(
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        features = self.shared(x)
        win_logits = self.head_winner(features)
        met_logits = self.head_method(features)
        return win_logits, met_logits

# ── Carregamento e Preparação ───────────────────────────────────────
def main():
    print(f"Carregando dados de {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    df['event_date'] = pd.to_datetime(df['event_date'])
    
    # Filtro Estreantes
    df = df[(df['a_total_fights'] > 0) & (df['b_total_fights'] > 0)].copy()
    df = df.sort_values('event_date', ascending=True).reset_index(drop=True)

    df['a_stance'] = df['a_stance'].map(STANCE_MAP).fillna(-1)
    df['b_stance'] = df['b_stance'].map(STANCE_MAP).fillna(-1)

    features_a = [f"a_{c}" for c in NUMERIC_BASE] + ['a_stance']
    features_b = [f"b_{c}" for c in NUMERIC_BASE] + ['b_stance']
    cols = features_a + features_b

    # Split Temporal
    n = len(df)
    n_train, n_val = int(n * 0.7), int(n * 0.2)
    
    df_train = df.iloc[:n_train].copy()
    df_val   = df.iloc[n_train:n_train+n_val].copy()
    df_test  = df.iloc[n_train+n_val:].copy()

    # Scaler & Medianas calculadas apenas no Treino
    X_tr_raw = df_train[cols].values
    col_medians = np.nanmedian(X_tr_raw, axis=0)
    
    def process_split(df_split):
        X = df_split[cols].values
        # Preencher NaNs com a mediana
        X[np.isnan(X)] = np.take(col_medians, np.where(np.isnan(X))[1])
        
        y_win = df_split['label'].values.astype(np.float32)
        y_met = df_split['method_class'].values.astype(np.int64)
        
        # Augmentação por Espelhamento
        X_mirror = np.concatenate([X[:, len(features_a):], X[:, :len(features_a)]], axis=1)
        y_win_mirror = 1.0 - y_win 
        y_met_mirror = y_met # O método da luta é o mesmo, independente do lado
        
        X_aug = np.vstack([X, X_mirror])
        y_win_aug = np.concatenate([y_win, y_win_mirror])
        y_met_aug = np.concatenate([y_met, y_met_mirror])
        return X_aug, y_win_aug, y_met_aug

    X_tr, y_win_tr, y_met_tr = process_split(df_train)
    X_va, y_win_va, y_met_va = process_split(df_val)
    X_te, y_win_te, y_met_te = process_split(df_test)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_va = scaler.transform(X_va)
    X_te = scaler.transform(X_te)

    def make_loader(X, yw, ym, shuffle=False):
        dataset = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(yw).unsqueeze(1), torch.LongTensor(ym))
        return DataLoader(dataset, batch_size=64, shuffle=shuffle)

    train_loader = make_loader(X_tr, y_win_tr, y_met_tr, shuffle=True)
    val_loader   = make_loader(X_va, y_win_va, y_met_va)
    test_loader  = make_loader(X_te, y_win_te, y_met_te)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskFightMLP(X_tr.shape[1]).to(device)
    
    criterion_win = nn.BCEWithLogitsLoss()
    criterion_met = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_val_loss = float('inf')
    patience, patience_counter = 20, 0

    print("Iniciando Treinamento Multi-Task...")
    for ep in range(1, 151):
        model.train()
        train_loss = 0.0
        for inputs, labels_win, labels_met in train_loader:
            inputs, labels_win, labels_met = inputs.to(device), labels_win.to(device), labels_met.to(device)
            optimizer.zero_grad()
            
            out_win, out_met = model(inputs)
            loss_win = criterion_win(out_win, labels_win)
            loss_met = criterion_met(out_met, labels_met)
            
            # Peso de 0.5 na loss do método para focar primariamente em descobrir quem ganha
            loss = loss_win + (0.5 * loss_met)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels_win, labels_met in val_loader:
                inputs, labels_win, labels_met = inputs.to(device), labels_win.to(device), labels_met.to(device)
                out_win, out_met = model(inputs)
                
                l_win = criterion_win(out_win, labels_win)
                l_met = criterion_met(out_met, labels_met)
                val_loss += (l_win + 0.5 * l_met).item() * inputs.size(0)
        val_loss /= len(val_loader.dataset)

        if ep % 10 == 1:
            print(f"  Ep {ep:>3} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{OUT_DIR}/best_model.pth")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping na época {ep}")
                break

# ── Avaliação Final no Teste ────────────────────────────────────
    model.load_state_dict(torch.load(f"{OUT_DIR}/best_model.pth", weights_only=True))
    model.eval()

    all_probs_win, all_labels_win = [], []
    all_preds_met, all_labels_met = [], []
    
    with torch.no_grad():
        for inputs, labels_win, labels_met in test_loader:
            inputs = inputs.to(device)
            out_win, out_met = model(inputs)
            
            # Pega as probabilidades brutas para calcular Brier, AUC, etc.
            probs_win = torch.sigmoid(out_win).cpu().numpy()
            preds_met = torch.argmax(out_met, dim=1).cpu().numpy()

            all_probs_win.extend(probs_win)
            all_labels_win.extend(labels_win.numpy())
            all_preds_met.extend(preds_met)
            all_labels_met.extend(labels_met.numpy())

    # Formatação dos arrays para as métricas do Scikit-Learn
    all_probs_win = np.array(all_probs_win).flatten()
    all_labels_win = np.array(all_labels_win).flatten()
    all_preds_win = (all_probs_win >= 0.5).astype(int)
    
    all_preds_met = np.array(all_preds_met).flatten()
    all_labels_met = np.array(all_labels_met).flatten()

    # Cálculo das métricas do Vencedor (Cabeça 1)
    acc = accuracy_score(all_labels_win, all_preds_win)
    auc = roc_auc_score(all_labels_win, all_probs_win)
    ll  = log_loss(all_labels_win, all_probs_win)
    bs  = brier_score_loss(all_labels_win, all_probs_win)

    print("\n══════════════════════════════════════════════════════════════")
    print(" 📊 AVALIAÇÃO FINAL — Conjunto de TESTE (nunca visto antes)")
    print("══════════════════════════════════════════════════════════════")
    print(" 🥊 CABEÇA 1: PREVISÃO DO VENCEDOR")
    print(f" Acurácia Global : {acc:.4f}  ({acc:.1%})")
    print(f" ROC-AUC         : {auc:.4f}")
    print(f" Log-Loss        : {ll:.4f}")
    print(f" Brier Score     : {bs:.4f}  (0=perfeito, 0.25=aleatório)\n")
    
    print(classification_report(all_labels_win, all_preds_win, target_names=["B vence", "A vence"]))

    print("\n 🎯 CABEÇA 2: PREVISÃO DO MÉTODO DE VITÓRIA")
    print(f" Acurácia Método : {accuracy_score(all_labels_met, all_preds_met):.1%}\n")
    print(classification_report(all_labels_met, all_preds_met, target_names=["Decisão", "Nocaute", "Finalização"]))

    print("\n 📈 CONFIANÇA VS ACURÁCIA (Probabilidade do Vencedor):")
    n_total = len(all_probs_win)
    for thr in [0.50, 0.55, 0.60, 0.65, 0.70]:
        mask = np.abs(all_probs_win - 0.5) >= (thr - 0.5)
        n_m = mask.sum()
        if n_m == 0:
            continue
        
        acc_t = accuracy_score(all_labels_win[mask], all_preds_win[mask])
        ci_lo = binom.interval(0.95, n_m, acc_t)[0] / n_m
        ci_hi = binom.interval(0.95, n_m, acc_t)[1] / n_m

        flag = "  ⚠️  (n pequeno)" if n_m < 50 else ""
        print(f"    |P − 0.5| ≥ {thr-0.5:.2f}  →  {n_m:>4} amostras  "
              f"acc={acc_t:.1%}  [IC95%: {ci_lo:.1%}–{ci_hi:.1%}]{flag}")

    # ── Salvar Artefatos ────────────────────────────────────────────
    with open(f"{OUT_DIR}/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    np.save(f"{OUT_DIR}/col_medians.npy", col_medians)
    
    print("\n✅ Todos os artefatos Multi-Task atualizados na pasta 'artifacts'.")

if __name__ == "__main__":
    main()