"""
UFC Fight Method Predictor — train_method.py
==============================================
Modelo SECUNDÁRIO: dado que uma luta acontece, prevê o MÉTODO de vitória
condicionado a "este lutador vence" (KO/TKO, Submissão, Decisão).

Este script NÃO prevê quem vence — isso continua sendo feito pelo FightMLP
já existente (train.py / artifacts/best_model.pt), que já foi testado e
calibrado. Não tocamos nele.
"""

import os, re, json, pickle, warnings
from typing import Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss, classification_report
warnings.filterwarnings("ignore")

# ── Reaproveita a engenharia de features e o split de train.py ────────────
from train import (
    NUMERIC_BASE, STANCE_MAP,
    load_and_clean, temporal_split, compute_streaks,
    df_row_to_dicts, build_row_features,
    make_loader, train_epoch, calibrate_temperature,
)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DATA_PATH = "../scripts/global_distribuido/ufc_dataset_male_only.csv"
OUT_DIR   = "artifacts_method"
os.makedirs(OUT_DIR, exist_ok=True)

METHOD_NAMES = ["KO/TKO", "Submissão", "Decisão"]


# ══════════════════════════════════════════════════════════════════════════
# 1. Categorização do método
# ══════════════════════════════════════════════════════════════════════════

def categorize_method(method_str) -> Optional[int]:
    if pd.isna(method_str):
        return None
    m = str(method_str)
    if re.search(r"KO|TKO", m, re.I):
        return 0
    if re.search(r"Sub", m, re.I):
        return 1
    if re.search(r"Dec", m, re.I):
        return 2
    return None


# ══════════════════════════════════════════════════════════════════════════
# 2. Construção de X, y — vencedor real sempre no slot A
# ══════════════════════════════════════════════════════════════════════════

def build_Xy_method(df: pd.DataFrame, col_medians=None):
    X_list, y_list = [], []
    dropped = 0

    for _, row in df.iterrows():
        y_cls = categorize_method(row["method"])
        if y_cls is None:
            dropped += 1
            continue

        sa, sb = df_row_to_dicts(row)
        a_venceu = bool(row["label"])
        winner_stats, loser_stats = (sa, sb) if a_venceu else (sb, sa)

        # Extrai os streaks calculados por compute_streaks (mesma fonte usada
        # pelo modelo de vencedor) e reordena para vencedor/perdedor, igual
        # já é feito acima com winner_stats/loser_stats
        aws = float(row.get("a_win_streak", 0))
        als = float(row.get("a_loss_streak", 0))
        bws = float(row.get("b_win_streak", 0))
        bls = float(row.get("b_loss_streak", 0))
        w_ws, w_ls, l_ws, l_ls = (aws, als, bws, bls) if a_venceu else (bws, bls, aws, als)

        X_list.append(build_row_features(winner_stats, loser_stats, w_ws, w_ls, l_ws, l_ls))
        y_list.append(y_cls)

    print(f"  Lutas descartadas (método fora de KO/Sub/Dec, ex. DQ): {dropped}")

    X = np.vstack(X_list)
    y = np.array(y_list, dtype=np.int64)

    X = np.where(np.isfinite(X), X, np.nan)
    if col_medians is None:
        col_medians = np.nanmedian(X, axis=0)
    nan_mask = np.isnan(X)
    X[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])

    return X, y, col_medians


# ══════════════════════════════════════════════════════════════════════════
# 3. Arquitetura
# ══════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════
# 4. Avaliação multi-classe
# ══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def eval_model_multiclass(model, loader, crit, device, ts=None):
    model.eval()
    total, preds, trues, all_probs = 0, [], [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        if ts is not None:
            logits = ts(logits)
        total += crit(logits, yb).item() * len(xb)
        probs = torch.softmax(logits, dim=1)
        preds.extend(logits.argmax(1).cpu().numpy())
        trues.extend(yb.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
    return (total / len(loader.dataset), np.array(preds), np.array(trues),
            np.array(all_probs))


# ══════════════════════════════════════════════════════════════════════════
# 5. Pipeline principal
# ══════════════════════════════════════════════════════════════════════════

def main():
    df = load_and_clean(DATA_PATH)
    df = compute_streaks(df)          # ← FIX: precisa rodar antes do split, igual em train.py
    df_train, df_val, df_test = temporal_split(df)

    print("\nConstruindo dataset de método (vencedor sempre no slot A)...")
    X_tr, y_tr, col_medians = build_Xy_method(df_train, col_medians=None)
    X_val, y_val, _         = build_Xy_method(df_val,   col_medians=col_medians)
    X_te,  y_te,  _         = build_Xy_method(df_test,  col_medians=col_medians)

    print(f"\n  Treino: {len(X_tr)}  Val: {len(X_val)}  Teste: {len(X_te)}")
    for cls, nome in enumerate(METHOD_NAMES):
        n_cls = int((y_tr == cls).sum())
        print(f"    Treino — {nome:<10}: {n_cls:>5}  ({n_cls/len(y_tr):.1%})")

    class_counts   = np.bincount(y_tr, minlength=3)
    class_weights  = class_counts.sum() / (3 * class_counts)
    class_weights  = torch.tensor(class_weights, dtype=torch.float32)
    print(f"  Pesos de classe (inverso da frequência): "
          f"{dict(zip(METHOD_NAMES, class_weights.tolist()))}")

    scaler = StandardScaler()
    X_tr  = scaler.fit_transform(X_tr)
    X_val = scaler.transform(X_val)
    X_te  = scaler.transform(X_te)

    tr_loader  = make_loader(X_tr,  y_tr,  shuffle=True,  bs=64)
    val_loader = make_loader(X_val, y_val, shuffle=False, bs=256)
    te_loader  = make_loader(X_te,  y_te,  shuffle=False, bs=256)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    model     = MethodMLP(X_tr.shape[1]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=2e-3)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=5e-4, epochs=200, steps_per_epoch=len(tr_loader))

    EPOCHS, PATIENCE = 200, 40
    best_val, no_improv = float("inf"), 0

    print(f"\n{'Ep':>4}  {'Train':>8}  {'Val':>8}  {'Val Acc':>8}")
    print("─" * 38)
    for ep in range(1, EPOCHS + 1):
        tr_l = train_epoch(model, tr_loader, optimizer, criterion, device)
        scheduler.step()
        val_l, preds, trues, _ = eval_model_multiclass(model, val_loader, criterion, device)
        acc = accuracy_score(trues, preds)

        if ep % 20 == 0 or ep == 1:
            print(f"{ep:>4}  {tr_l:>8.4f}  {val_l:>8.4f}  {acc:>7.1%}")

        if val_l < best_val - 1e-5:
            best_val, no_improv = val_l, 0
            torch.save(model.state_dict(), f"{OUT_DIR}/best_model.pt")
        else:
            no_improv += 1
            if no_improv >= PATIENCE:
                print(f"\n  Early stopping na época {ep} (melhor val_loss: {best_val:.4f})")
                break

    model.load_state_dict(torch.load(f"{OUT_DIR}/best_model.pt", map_location=device))
    print("\nCalibrando temperatura com conjunto de validação...")
    ts = calibrate_temperature(model, val_loader, device)
    torch.save(ts.state_dict(), f"{OUT_DIR}/temperature.pt")

    # ── Avaliação final no TESTE ─────────────────────────────────────────────
    _, te_preds, te_trues, te_probs = eval_model_multiclass(
        model, te_loader, criterion, device, ts=ts)

    acc = accuracy_score(te_trues, te_preds)
    ll = log_loss(te_trues, te_probs, labels=[0, 1, 2])

    print("\n" + "═" * 62)
    print("  📊 AVALIAÇÃO FINAL (método) — Conjunto de TESTE (nunca visto antes)")
    print("═" * 62)
    print(f"  Lutas no teste : {len(y_te)}")
    print(f"  Acurácia       : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"  Log-Loss       : {ll:.4f}")
    print()
    print(classification_report(te_trues, te_preds, target_names=METHOD_NAMES))

    # ── NOVA FUNCIONALIDADE: Confiança vs Acurácia por Método ────────────────
    from scipy.stats import binom
    
    print("  Confiança vs Acurácia (Por Método Previsto):")
    for m_idx, m_name in enumerate(METHOD_NAMES):
        print(f"\n  ➤ Quando o modelo aposta em: {m_name}")
        
        mask_m = (te_preds == m_idx)
        
        if mask_m.sum() == 0:
            print("    Nenhuma previsão para este método no conjunto de teste.")
            continue
            
        probs_m = te_probs[mask_m, m_idx]
        trues_m = te_trues[mask_m]
        
        for thr in [0.20, 0.25, 0.30, 0.35, 0.40]:
            mask_thr = probs_m >= thr
            n_m = mask_thr.sum()
            
            if n_m == 0:
                print(f"    Prob ≥ {thr:.2f}  →     0 amostras")
                continue
                
            correct = (trues_m[mask_thr] == m_idx).sum()
            acc_t = correct / n_m
            
            ci_lo = binom.interval(0.95, n_m, acc_t)[0] / n_m
            ci_hi = binom.interval(0.95, n_m, acc_t)[1] / n_m
            
            flag = "  ⚠️  (n pequeno)" if n_m < 50 else ""
            print(f"    Prob ≥ {thr:.2f}  →  {n_m:>4} amostras  "
                  f"acc real={acc_t:.1%}  [IC95%: {ci_lo:.1%}–{ci_hi:.1%}]{flag}")
    
    print("\n" + "═" * 62)

    # ── Salvar artefatos ─────────────────────────────────────────────────────
    with open(f"{OUT_DIR}/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    np.save(f"{OUT_DIR}/col_medians.npy", col_medians)

    meta = {
        "n_features":   int(X_tr.shape[1]),
        "numeric_base": NUMERIC_BASE,
        "stance_map":   STANCE_MAP,
        "method_names": METHOD_NAMES,
        "class_weights": class_weights.tolist(),
        "test_acc":     float(acc),
        "test_logloss": float(ll),
    }
    with open(f"{OUT_DIR}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✅ Artefatos salvos em: {OUT_DIR}")

if __name__ == "__main__":
    main()