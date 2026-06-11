"""
UFC Fight Outcome & Method Predictor — train_v5.py
====================================================
Combina o melhor do v3 e do v4:

HERDADO DO v3:
  ✅ 105 features: deltas (A−B), absolutos, derivadas, streaks, stance one-hot
  ✅ Split temporal 80/10/10 com verificação de sobreposição
  ✅ OneCycleLR scheduler
  ✅ Calibração de temperatura (Temperature Scaling)
  ✅ Augmentação por espelhamento por split
  ✅ Mediana e Scaler calculados só no treino

HERDADO DO v4:
  ✅ Arquitetura Multi-Task (cabeça de vencedor + cabeça de método)
  ✅ Predição simétrica na inferência

NOVIDADES DO v5:
  ✅ CrossEntropyLoss com class weights para método (corrige colapso em "Decisão")
  ✅ Loss weighting dinâmico: peso do método cresce gradualmente após warm-up
     → protege a acurácia do vencedor enquanto o método aprende
  ✅ Dataset: ufc_dataset_v5.csv (v4 + colunas de streak calculadas aqui)
  ✅ Predição simétrica separada por cabeça (vencedor subtrai, método soma)
  ✅ Relatório de avaliação unificado com análise por threshold de confiança
"""

import os, json, pickle, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, roc_auc_score, log_loss,
    brier_score_loss, classification_report,
)
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from scipy.stats import binom
warnings.filterwarnings("ignore")

# ── Reprodutibilidade ──────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Configuração ───────────────────────────────────────────────────────────────
DATA_PATH = "ufc_dataset_v4.csv"   # v4 com method_class — streaks calculados aqui
OUT_DIR   = "artifacts"
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

# Mapeamento de método: 0=Decisão, 1=Nocaute/TKO, 2=Finalização
METHOD_NAMES = ["Decisão", "Nocaute", "Finalização"]


# ══════════════════════════════════════════════════════════════════════════════
# 1. Carregamento, limpeza e streaks
# ══════════════════════════════════════════════════════════════════════════════

def compute_streaks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula win_streak e loss_streak para cada lutador antes de cada luta.
    Idêntico ao v3 — processamento em ordem cronológica, sem data leakage.
    """
    def _streak(history: list) -> tuple[int, int]:
        if not history:
            return 0, 0
        win_streak = loss_streak = 0
        if history[-1]:
            for won in reversed(history):
                if won:  win_streak  += 1
                else:    break
        else:
            for won in reversed(history):
                if not won: loss_streak += 1
                else:       break
        return win_streak, loss_streak

    fighter_history: dict[str, list[bool]] = {}
    rows = {
        "a_win_streak":  [],
        "a_loss_streak": [],
        "b_win_streak":  [],
        "b_loss_streak": [],
    }

    for _, row in df.iterrows():
        fa, fb = row["fighter_a"], row["fighter_b"]
        wa, la = _streak(fighter_history.get(fa, []))
        wb, lb = _streak(fighter_history.get(fb, []))

        rows["a_win_streak"].append(wa)
        rows["a_loss_streak"].append(la)
        rows["b_win_streak"].append(wb)
        rows["b_loss_streak"].append(lb)

        a_won = bool(row["label"])
        fighter_history.setdefault(fa, []).append(a_won)
        fighter_history.setdefault(fb, []).append(not a_won)

    df = df.copy()
    for col, vals in rows.items():
        df[col] = vals

    print(f"\nStreaks calculados:")
    print(f"  win_streak  — max: {df['a_win_streak'].max():>2}  "
          f"média: {df['a_win_streak'].mean():.2f}")
    print(f"  loss_streak — max: {df['a_loss_streak'].max():>2}  "
          f"média: {df['a_loss_streak'].mean():.2f}")
    return df


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["event_date"] = pd.to_datetime(df["event_date"])
    df = df.sort_values("event_date").reset_index(drop=True)

    before = len(df)
    mask   = (df["a_total_fights"] > 0) & (df["b_total_fights"] > 0)
    df     = df[mask].reset_index(drop=True)

    removed = before - len(df)
    print(f"Lutas originais     : {before}")
    print(f"Estreantes removidos: {removed}")
    print(f"Lutas após filtro   : {len(df)}")

    # Verificar coluna de método
    assert "method_class" in df.columns, \
        "ERRO: coluna 'method_class' não encontrada. Use ufc_dataset_v5.csv (baseado no v4)."
    dist = df["method_class"].value_counts().sort_index()
    print(f"\nDistribuição do método:")
    for i, name in enumerate(METHOD_NAMES):
        n   = dist.get(i, 0)
        pct = n / len(df) * 100
        print(f"  {name:<12}: {n:>4} ({pct:.1f}%)")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. Split temporal
# ══════════════════════════════════════════════════════════════════════════════

def temporal_split(df: pd.DataFrame):
    n       = len(df)
    n_train = int(n * 0.80)
    n_val   = int(n * 0.10)

    df_train = df.iloc[:n_train].reset_index(drop=True)
    df_val   = df.iloc[n_train : n_train + n_val].reset_index(drop=True)
    df_test  = df.iloc[n_train + n_val :].reset_index(drop=True)

    print(f"\nSplit temporal:")
    print(f"  Treino : {len(df_train):>4} lutas  "
          f"[{df_train['event_date'].min().date()} → {df_train['event_date'].max().date()}]")
    print(f"  Val    : {len(df_val):>4} lutas  "
          f"[{df_val['event_date'].min().date()} → {df_val['event_date'].max().date()}]")
    print(f"  Teste  : {len(df_test):>4} lutas  "
          f"[{df_test['event_date'].min().date()} → {df_test['event_date'].max().date()}]")

    assert df_train["event_date"].max() <= df_val["event_date"].min(),  \
        "ERRO: treino e val se sobrepõem!"
    assert df_val["event_date"].max()   <= df_test["event_date"].min(), \
        "ERRO: val e teste se sobrepõem!"
    print("  ✅ Sem sobreposição de datas entre splits")

    return df_train, df_val, df_test


# ══════════════════════════════════════════════════════════════════════════════
# 3. Feature engineering (idêntico ao v3, agora inclui streaks)
# ══════════════════════════════════════════════════════════════════════════════

# Adicionamos streaks ao NUMERIC_BASE para o v5
NUMERIC_BASE_V5 = NUMERIC_BASE + ["win_streak", "loss_streak"]

def _derived(stats: dict) -> list:
    """8 features derivadas por lutador."""
    wins   = float(stats.get("wins",   0) or 0)
    total  = float(stats.get("total_fights", 0) or 0)
    ko     = float(stats.get("wins_ko",  0) or 0)
    sub    = float(stats.get("wins_sub", 0) or 0)
    dec    = float(stats.get("wins_dec", 0) or 0)
    losses = float(stats.get("losses",   0) or 0)
    td_15  = float(stats.get("td_per_15min",  0) or 0)
    sub_15 = float(stats.get("sub_per_15min", 0) or 0)

    return [
        wins / total        if total > 0 else 0.5,   # win_rate
        (ko + sub) / wins   if wins  > 0 else 0.0,   # finish_rate
        ko  / wins          if wins  > 0 else 0.0,   # ko_rate
        sub / wins          if wins  > 0 else 0.0,   # sub_rate
        dec / wins          if wins  > 0 else 0.0,   # dec_rate
        losses / total      if total > 0 else 0.5,   # loss_rate
        float(np.log1p(total)),                       # exp_score (log da experiência)
        td_15 + sub_15 * 0.5,                         # grapple_score
    ]


def build_row_features(row_a: dict, row_b: dict) -> np.ndarray:
    """
    Constrói vetor de ~109 features:
      1. Deltas (A − B) para NUMERIC_BASE_V5        → 27 features
      2. Absolutos de A                              → 27 features
      3. Absolutos de B                              → 27 features
      4. Features derivadas de A (8) + B (8) + delta → 24 features
      5. Stance one-hot de A (3) + B (3)             →  6 features
      6. Matchup de stance (3 features)              →  3 features
    Total: 114 features
    """
    feats = []

    # 1. Deltas numéricos (A − B)
    for col in NUMERIC_BASE_V5:
        va = float(row_a.get(col, 0) or 0)
        vb = float(row_b.get(col, 0) or 0)
        feats.append(va - vb)

    # 2 & 3. Absolutos de A, depois B
    for stats in (row_a, row_b):
        for col in NUMERIC_BASE_V5:
            feats.append(float(stats.get(col, 0) or 0))

    # 4. Derivadas de A, B e deltas
    da = _derived(row_a)
    db = _derived(row_b)
    feats.extend(da)
    feats.extend(db)
    feats.extend([a - b for a, b in zip(da, db)])

    # 5. Stance one-hot × 2
    for stats in (row_a, row_b):
        idx = STANCE_MAP.get(str(stats.get("stance", "Orthodox")), 0)
        ohe = [0.0, 0.0, 0.0]
        ohe[idx] = 1.0
        feats.extend(ohe)

    # 6. Matchup de stance
    sa = STANCE_MAP.get(str(row_a.get("stance", "Orthodox")), 0)
    sb = STANCE_MAP.get(str(row_b.get("stance", "Orthodox")), 0)
    feats.append(float(sa != sb))   # stances diferentes?
    feats.append(float(sa == 1))    # A é southpaw?
    feats.append(float(sb == 1))    # B é southpaw?

    return np.array(feats, dtype=np.float32)


def df_row_to_dicts(row: pd.Series):
    """Converte linha do DataFrame em dois dicts (stats_a, stats_b)."""
    stats_a = {c[2:]: row[c] for c in row.index if c.startswith("a_")}
    stats_b = {c[2:]: row[c] for c in row.index if c.startswith("b_")}
    return stats_a, stats_b


def build_Xy(df: pd.DataFrame, col_medians=None):
    """
    Constrói X (features), y_win (label vencedor) e y_met (label método)
    com augmentação por espelhamento.
    - col_medians=None → calculado aqui (apenas para treino)
    - col_medians=array → usa os valores passados (val/teste)
    """
    X_orig,  X_mirr  = [], []
    yw_orig, yw_mirr = [], []
    ym_orig, ym_mirr = [], []

    for _, row in df.iterrows():
        label  = int(row["label"])
        method = int(row["method_class"])
        sa, sb = df_row_to_dicts(row)

        X_orig.append(build_row_features(sa, sb))
        yw_orig.append(label)
        ym_orig.append(method)

        # Espelho: A↔B, label de vencedor inverte, método NÃO inverte
        X_mirr.append(build_row_features(sb, sa))
        yw_mirr.append(1 - label)
        ym_mirr.append(method)

    X   = np.vstack(X_orig + X_mirr)
    y_w = np.array(yw_orig + yw_mirr, dtype=np.int64)
    y_m = np.array(ym_orig + ym_mirr, dtype=np.int64)

    # Substitui inf → nan para imputação com mediana
    X = np.where(np.isfinite(X), X, np.nan)

    if col_medians is None:
        col_medians = np.nanmedian(X, axis=0)

    nan_mask = np.isnan(X)
    X[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])

    return X, y_w, y_m, col_medians


# ══════════════════════════════════════════════════════════════════════════════
# 4. Arquitetura Multi-Task
# ══════════════════════════════════════════════════════════════════════════════

class FightMLPv5(nn.Module):
    """
    Tronco compartilhado 512→256→128 + duas cabeças independentes:
      - head_winner: binária (B vence / A vence)
      - head_method: 3 classes (Decisão / KO / Sub)

    Dropout mais alto no tronco (aprendizado robusto),
    menor nas cabeças (especialização suave).
    """
    def __init__(self, n_in: int):
        super().__init__()

        # Tronco compartilhado
        self.shared = nn.Sequential(
            nn.Linear(n_in, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.40),
            nn.Linear(512, 256),  nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.35),
            nn.Linear(256, 128),  nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.25),
        )

        # Cabeça 1: Vencedor (2 classes para usar CrossEntropy igual ao v3)
        self.head_winner = nn.Sequential(
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(64, 2),
        )

        # Cabeça 2: Método (3 classes)
        self.head_method = nn.Sequential(
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(64, 3),
        )

    def forward(self, x):
        shared = self.shared(x)
        return self.head_winner(shared), self.head_method(shared)


class TemperatureScaler(nn.Module):
    """Calibração pós-treinamento para a cabeça do vencedor."""
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        return logits / self.temperature


# ══════════════════════════════════════════════════════════════════════════════
# 5. Helpers de treino e avaliação
# ══════════════════════════════════════════════════════════════════════════════

def make_loader(X, y_w, y_m, shuffle=False, bs=64):
    ds = TensorDataset(
        torch.FloatTensor(X),
        torch.LongTensor(y_w),
        torch.LongTensor(y_m),
    )
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, drop_last=shuffle)


def method_weight(y_m_train: np.ndarray, device) -> torch.Tensor:
    """
    Calcula peso inversamente proporcional à frequência de cada classe de método.
    Isso corrige o colapso em 'Decisão' que ocorreu no v4.
    """
    counts = np.bincount(y_m_train, minlength=3).astype(float)
    total  = counts.sum()
    # w_i = total / (n_classes * count_i)  — fórmula padrão de class weighting
    w = total / (3.0 * counts)
    w = w / w.sum() * 3.0   # renormaliza para soma=3 (escala neutra)
    print(f"\nClass weights para método:")
    for i, name in enumerate(METHOD_NAMES):
        print(f"  {name:<12}: {w[i]:.3f}  (n={int(counts[i])})")
    return torch.tensor(w, dtype=torch.float32).to(device)


def dynamic_method_weight(epoch: int, warmup: int = 30, max_w: float = 0.4) -> float:
    """
    Peso dinâmico da loss de método:
      - Epochs 1..warmup: cresce linearmente de 0.05 até max_w
      - Após warmup: constante em max_w

    Isso garante que o tronco aprenda primeiro o vencedor,
    depois divide capacidade com o método.
    max_w=0.4 é menor que os 0.5 do v4, priorizando o vencedor.
    """
    if epoch <= warmup:
        return 0.05 + (max_w - 0.05) * (epoch / warmup)
    return max_w


def train_epoch(model, loader, opt, crit_win, crit_met, device, epoch):
    model.train()
    total_loss = 0.0
    w_met = dynamic_method_weight(epoch)

    for xb, yw, ym in loader:
        xb, yw, ym = xb.to(device), yw.to(device), ym.to(device)
        opt.zero_grad()

        logits_w, logits_m = model(xb)
        loss_win = crit_win(logits_w, yw)
        loss_met = crit_met(logits_m, ym)
        loss     = loss_win + w_met * loss_met

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total_loss += loss.item() * len(xb)

    return total_loss / len(loader.dataset), w_met


@torch.no_grad()
def eval_model(model, loader, crit_win, crit_met, device, ts=None, w_met=0.4):
    model.eval()
    total_loss = 0.0
    preds_w, trues_w, probs_w = [], [], []
    preds_m, trues_m           = [], []

    for xb, yw, ym in loader:
        xb, yw, ym = xb.to(device), yw.to(device), ym.to(device)
        logits_w, logits_m = model(xb)

        if ts is not None:
            logits_w = ts(logits_w)

        l_win = crit_win(logits_w, yw)
        l_met = crit_met(logits_m, ym)
        total_loss += (l_win + w_met * l_met).item() * len(xb)

        p_w = torch.softmax(logits_w, dim=1)[:, 1]
        probs_w.extend(p_w.cpu().numpy())
        preds_w.extend(logits_w.argmax(1).cpu().numpy())
        trues_w.extend(yw.cpu().numpy())

        preds_m.extend(logits_m.argmax(1).cpu().numpy())
        trues_m.extend(ym.cpu().numpy())

    n = len(loader.dataset)
    return (
        total_loss / n,
        np.array(preds_w), np.array(trues_w), np.array(probs_w),
        np.array(preds_m), np.array(trues_m),
    )


def calibrate_temperature(model, val_loader, device):
    """Temperature Scaling — calibra apenas a cabeça do vencedor."""
    ts  = TemperatureScaler().to(device)
    opt = torch.optim.LBFGS([ts.temperature], lr=0.01, max_iter=100)
    crit = nn.CrossEntropyLoss()

    all_logits, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for xb, yw, _ in val_loader:
            logits_w, _ = model(xb.to(device))
            all_logits.append(logits_w.cpu())
            all_labels.append(yw)

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


# ══════════════════════════════════════════════════════════════════════════════
# 6. Avaliação final e plots
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_and_report(
    preds_w, trues_w, probs_w,
    preds_m, trues_m,
    df_test,
):
    acc_w = accuracy_score(trues_w, preds_w)
    auc   = roc_auc_score(trues_w, probs_w)
    ll    = log_loss(trues_w, probs_w)
    bs    = brier_score_loss(trues_w, probs_w)
    acc_m = accuracy_score(trues_m, preds_m)

    print("\n" + "═" * 62)
    print("  📊 AVALIAÇÃO FINAL — Conjunto de TESTE (nunca visto antes)")
    print("═" * 62)
    print(f"  Lutas no teste  : {len(df_test)} ({len(preds_w)} c/ augmentação)")
    print(f"  Período         : {df_test['event_date'].min().date()} "
          f"→ {df_test['event_date'].max().date()}")

    print(f"\n  🥊 CABEÇA 1: VENCEDOR")
    print(f"  Acurácia   : {acc_w:.4f}  ({acc_w*100:.1f}%)")
    print(f"  ROC-AUC    : {auc:.4f}")
    print(f"  Log-Loss   : {ll:.4f}")
    print(f"  Brier Score: {bs:.4f}  (0=perfeito, 0.25=aleatório)\n")
    print(classification_report(trues_w, preds_w, target_names=["B vence", "A vence"]))

    print(f"  🎯 CABEÇA 2: MÉTODO DE VITÓRIA")
    print(f"  Acurácia   : {acc_m:.4f}  ({acc_m*100:.1f}%)\n")
    print(classification_report(trues_m, preds_m, target_names=METHOD_NAMES))

    print("  📈 CONFIANÇA VS ACURÁCIA (Vencedor):")
    for thr in [0.50, 0.55, 0.60, 0.65, 0.70]:
        mask = (probs_w >= thr) | (probs_w <= 1 - thr)
        n_m  = mask.sum()
        if n_m == 0:
            print(f"    |P−0.5| ≥ {thr-0.5:.2f}  →  modelo nunca atinge essa confiança")
            break
        correct = ((probs_w[mask] >= 0.5) == trues_w[mask].astype(bool)).sum()
        acc_t   = correct / n_m
        ci_lo   = binom.interval(0.95, n_m, acc_t)[0] / n_m
        ci_hi   = binom.interval(0.95, n_m, acc_t)[1] / n_m
        flag    = "  ⚠️  (n pequeno)" if n_m < 50 else ""
        print(f"    |P−0.5| ≥ {thr-0.5:.2f}  →  {n_m:>4} amostras  "
              f"acc={acc_t:.1%}  [IC95%: {ci_lo:.1%}–{ci_hi:.1%}]{flag}")

    return {"acc_winner": acc_w, "auc": auc, "logloss": ll,
            "brier": bs, "acc_method": acc_m}


def save_plots(history: dict, te_trues_w, te_preds_w,
               te_trues_m, te_preds_m, out_dir: str):
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("v5 — Curvas de Treinamento", fontsize=14, fontweight="bold")

    # Loss
    axes[0].plot(epochs, history["train_loss"], label="Treino",    color="royalblue")
    axes[0].plot(epochs, history["val_loss"],   label="Validação", color="tomato")
    axes[0].set_title("Loss combinada")
    axes[0].set_xlabel("Época")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Acurácia validação
    axes[1].plot(epochs, history["val_acc_w"], color="seagreen", label="Val Acc Vencedor")
    axes[1].axhline(y=max(history["val_acc_w"]), color="gray", linestyle="--",
                    alpha=0.6, label=f"Máx: {max(history['val_acc_w']):.1%}")
    axes[1].set_title("Acurácia — Vencedor (Validação)")
    axes[1].set_xlabel("Época")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Acurácia método
    axes[2].plot(epochs, history["val_acc_m"], color="darkorange", label="Val Acc Método")
    axes[2].axhline(y=max(history["val_acc_m"]), color="gray", linestyle="--",
                    alpha=0.6, label=f"Máx: {max(history['val_acc_m']):.1%}")
    axes[2].set_title("Acurácia — Método (Validação)")
    axes[2].set_xlabel("Época")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{out_dir}/training_curves.png", dpi=150)
    plt.close()
    print(f"  📈 Curvas salvas em: {out_dir}/training_curves.png")

    # Matriz de confusão — Vencedor
    cm   = confusion_matrix(te_trues_w, te_preds_w)
    disp = ConfusionMatrixDisplay(cm, display_labels=["B vence", "A vence"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("Confusão — Vencedor (Teste)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/confusion_winner.png", dpi=150)
    plt.close()

    # Matriz de confusão — Método
    cm_m   = confusion_matrix(te_trues_m, te_preds_m)
    disp_m = ConfusionMatrixDisplay(cm_m, display_labels=METHOD_NAMES)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp_m.plot(ax=ax, cmap="Oranges", values_format="d")
    ax.set_title("Confusão — Método (Teste)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/confusion_method.png", dpi=150)
    plt.close()
    print(f"  🔢 Matrizes salvas em: {out_dir}/")


# ══════════════════════════════════════════════════════════════════════════════
# 7. Pipeline principal
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Dados ────────────────────────────────────────────────────────────────
    df = load_and_clean(DATA_PATH)
    df = compute_streaks(df)
    df_train, df_val, df_test = temporal_split(df)

    # ── Features ─────────────────────────────────────────────────────────────
    X_tr,  y_win_tr,  y_met_tr,  col_medians = build_Xy(df_train)
    X_val, y_win_val, y_met_val, _           = build_Xy(df_val,  col_medians)
    X_te,  y_win_te,  y_met_te,  _           = build_Xy(df_test, col_medians)

    n_features = X_tr.shape[1]
    print(f"\nFeatures por amostra : {n_features}")
    print(f"Amostras treino (aug): {len(X_tr)}")
    print(f"Amostras val    (aug): {len(X_val)}")
    print(f"Amostras teste  (aug): {len(X_te)}")

    assert not np.isnan(X_tr).any()
    assert not np.isnan(X_val).any()
    assert not np.isnan(X_te).any()
    print("  ✅ Zero NaNs em todos os splits")

    # ── Normalização ─────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_tr  = scaler.fit_transform(X_tr)
    X_val = scaler.transform(X_val)
    X_te  = scaler.transform(X_te)

    # ── Loaders ──────────────────────────────────────────────────────────────
    tr_loader  = make_loader(X_tr,  y_win_tr,  y_met_tr,  shuffle=True,  bs=64)
    val_loader = make_loader(X_val, y_win_val, y_met_val, shuffle=False, bs=256)
    te_loader  = make_loader(X_te,  y_win_te,  y_met_te,  shuffle=False, bs=256)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # ── Modelo e critérios ────────────────────────────────────────────────────
    model    = FightMLPv5(n_features).to(device)
    w_class  = method_weight(y_met_tr, device)

    # Vencedor: CrossEntropy balanceada (classes já equilibradas pela augmentação)
    crit_win = nn.CrossEntropyLoss()
    # Método: CrossEntropy COM class weights para corrigir desbalanceamento
    crit_met = nn.CrossEntropyLoss(weight=w_class)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=2e-3)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=5e-4, epochs=200,
        steps_per_epoch=len(tr_loader),
    )

    # ── Loop de treinamento ───────────────────────────────────────────────────
    EPOCHS   = 200
    PATIENCE = 40
    best_val = float("inf")
    no_improv = 0
    history  = {
        "train_loss": [], "val_loss": [],
        "val_acc_w":  [], "val_acc_m": [],
        "w_met":      [],
    }

    print(f"\n{'Ep':>4}  {'Train':>8}  {'Val':>8}  "
          f"{'AccW':>7}  {'AccM':>7}  {'wMet':>6}")
    print("─" * 52)

    for ep in range(1, EPOCHS + 1):
        tr_loss, w_met = train_epoch(
            model, tr_loader, optimizer, crit_win, crit_met, device, ep)
        scheduler.step()

        val_loss, pv_w, tv_w, probs_v, pv_m, tv_m = eval_model(
            model, val_loader, crit_win, crit_met, device, w_met=0.4)

        acc_w = accuracy_score(tv_w, pv_w)
        acc_m = accuracy_score(tv_m, pv_m)

        history["train_loss"].append(float(tr_loss))
        history["val_loss"].append(float(val_loss))
        history["val_acc_w"].append(float(acc_w))
        history["val_acc_m"].append(float(acc_m))
        history["w_met"].append(float(w_met))

        if ep % 20 == 0 or ep == 1:
            print(f"{ep:>4}  {tr_loss:>8.4f}  {val_loss:>8.4f}  "
                  f"{acc_w:>6.1%}  {acc_m:>6.1%}  {w_met:>5.3f}")

        if val_loss < best_val - 1e-5:
            best_val   = val_loss
            no_improv  = 0
            torch.save(model.state_dict(), f"{OUT_DIR}/best_model.pt")
        else:
            no_improv += 1
            if no_improv >= PATIENCE:
                print(f"\n  Early stopping na época {ep} "
                      f"(melhor val_loss: {best_val:.4f})")
                break

    # ── Calibração de temperatura ─────────────────────────────────────────────
    model.load_state_dict(
        torch.load(f"{OUT_DIR}/best_model.pt", map_location=device))
    print("\nCalibrando temperatura com conjunto de validação...")
    ts = calibrate_temperature(model, val_loader, device)
    torch.save(ts.state_dict(), f"{OUT_DIR}/temperature.pt")

    # ── Avaliação final no TESTE ──────────────────────────────────────────────
    _, te_preds_w, te_trues_w, te_probs_w, te_preds_m, te_trues_m = eval_model(
        model, te_loader, crit_win, crit_met, device, ts=ts)

    metrics = evaluate_and_report(
        te_preds_w, te_trues_w, te_probs_w,
        te_preds_m, te_trues_m,
        df_test,
    )

    # ── Salvar artefatos ──────────────────────────────────────────────────────
    with open(f"{OUT_DIR}/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    np.save(f"{OUT_DIR}/col_medians.npy", col_medians)

    meta = {
        "version":      "v5",
        "n_features":   n_features,
        "numeric_base": NUMERIC_BASE_V5,
        "stance_map":   STANCE_MAP,
        "method_names": METHOD_NAMES,
        "history":      history,
        **metrics,
        "train_period": [str(df_train["event_date"].min().date()),
                         str(df_train["event_date"].max().date())],
        "val_period":   [str(df_val["event_date"].min().date()),
                         str(df_val["event_date"].max().date())],
        "test_period":  [str(df_test["event_date"].min().date()),
                         str(df_test["event_date"].max().date())],
    }
    with open(f"{OUT_DIR}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✅ Artefatos salvos em: {OUT_DIR}/")
    print("\nGerando gráficos...")
    save_plots(history, te_trues_w, te_preds_w,
               te_trues_m, te_preds_m, OUT_DIR)


if __name__ == "__main__":
    main()
