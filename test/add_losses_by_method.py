# add_losses_by_method.py
# ========================
# Adiciona colunas de derrotas por método (losses_ko, losses_sub, losses_dec)
# ao dataset existente, processando cronologicamente sem re-scraping.
#
# Uso:  python add_losses_by_method.py
#       (roda a partir da pasta test/)

import re
import pandas as pd

INPUT  = "../model/ufc_dataset.csv"
OUTPUT = "../model/ufc_dataset.csv"   # sobrescreve in-place

df = pd.read_csv(INPUT)
df["event_date"] = pd.to_datetime(df["event_date"])

# 1. Ordena do MAIS ANTIGO para o MAIS RECENTE
df = df.sort_values("event_date", ascending=True).reset_index(drop=True)

# 2. Dicionário de memória: { "Nome do Lutador": {"ko": N, "sub": N, "dec": N} }
fighter_losses = {}

# 3. Criar as novas colunas preenchidas com 0
for prefix in ["a_", "b_"]:
    df[f"{prefix}losses_ko"]  = 0
    df[f"{prefix}losses_sub"] = 0
    df[f"{prefix}losses_dec"] = 0

# 4. Helpers para categorizar método
def is_ko(method):
    return bool(re.search(r"KO|TKO", str(method), re.I))

def is_sub(method):
    return bool(re.search(r"Sub", str(method), re.I))

def is_dec(method):
    return bool(re.search(r"Dec", str(method), re.I))

# 5. Iterar luta por luta no tempo
for idx, row in df.iterrows():
    fa = row["fighter_a"]
    fb = row["fighter_b"]
    winner = row["winner"]      # A ou B
    method = str(row["method"]) # KO/TKO, Submission, Decision - ...

    # A. PREENCHER AS FEATURES (foto de ANTES da luta começar)
    for f, prefix in [(fa, "a_"), (fb, "b_")]:
        if f in fighter_losses:
            df.at[idx, f"{prefix}losses_ko"]  = fighter_losses[f]["ko"]
            df.at[idx, f"{prefix}losses_sub"] = fighter_losses[f]["sub"]
            df.at[idx, f"{prefix}losses_dec"] = fighter_losses[f]["dec"]

    # B. ATUALIZAR O HISTÓRICO (resultado desta luta para o futuro)
    # Inicializa se necessário
    if fa not in fighter_losses:
        fighter_losses[fa] = {"ko": 0, "sub": 0, "dec": 0}
    if fb not in fighter_losses:
        fighter_losses[fb] = {"ko": 0, "sub": 0, "dec": 0}

    if winner == "A":
        # B perdeu
        if is_ko(method):
            fighter_losses[fb]["ko"] += 1
        elif is_sub(method):
            fighter_losses[fb]["sub"] += 1
        elif is_dec(method):
            fighter_losses[fb]["dec"] += 1
    elif winner == "B":
        # A perdeu
        if is_ko(method):
            fighter_losses[fa]["ko"] += 1
        elif is_sub(method):
            fighter_losses[fa]["sub"] += 1
        elif is_dec(method):
            fighter_losses[fa]["dec"] += 1

# 6. Volta a ordenação para o padrão (Mais recente no topo)
df = df.sort_values("event_date", ascending=False).reset_index(drop=True)

# 7. Salva
df.to_csv(OUTPUT, index=False)

# 8. Diagnóstico
print(f"Dataset atualizado: {OUTPUT}")
print(f"Total de lutas: {len(df)}")
print(f"\nNovas colunas adicionadas:")
for col in ["a_losses_ko", "a_losses_sub", "a_losses_dec",
            "b_losses_ko", "b_losses_sub", "b_losses_dec"]:
    print(f"  {col}: min={df[col].min()}, max={df[col].max()}, "
          f"mean={df[col].mean():.2f}, zeros={int((df[col] == 0).sum())}")

# 9. Validação: lutadores com muitas lutas devem ter losses condizentes
top = df.groupby("fighter_a").agg(
    fights=("a_total_fights", "max"),
    losses_ko=("a_losses_ko", "max"),
    losses_sub=("a_losses_sub", "max"),
    losses_dec=("a_losses_dec", "max"),
).sort_values("fights", ascending=False).head(10)
print(f"\nTop 10 lutadores mais experientes (sanity check):")
print(top.to_string())
