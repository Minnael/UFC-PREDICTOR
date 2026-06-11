# gerar_historico.py
import pandas as pd

# 1. Carrega o dataset
df = pd.read_csv('ufc_dataset.csv')
df['event_date'] = pd.to_datetime(df['event_date'])

# 2. Ordena do MAIS ANTIGO para o MAIS RECENTE para o fluxo do tempo estar correto
df = df.sort_values('event_date', ascending=True).reset_index(drop=True)

# 3. Dicionário de memória: { "Nome do Lutador": [res_luta1, res_luta2, ...] }
fighter_history = {}

# 4. Criar as novas colunas preenchidas com -1.0 (significa "Ainda não tem histórico")
for prefix in ['a_', 'b_']:
    for i in range(1, 6):
        df[f'{prefix}last_fight_{i}'] = -1.0 

# 5. Iterar luta por luta no tempo
for idx, row in df.iterrows():
    fa = row['fighter_a']
    fb = row['fighter_b']
    winner = row['winner'] # A, B ou Draw

    # A. PREENCHER AS FEATURES (A foto de ANTES da luta começar)
    for f, prefix in [(fa, 'a_'), (fb, 'b_')]:
        if f in fighter_history:
            # Pega até as 5 últimas lutas e inverte para o índice 1 ser a luta mais recente
            hist = list(reversed(fighter_history[f][-5:]))
            for i, res in enumerate(hist):
                df.at[idx, f'{prefix}last_fight_{i+1}'] = res

    # B. ATUALIZAR O HISTÓRICO (O resultado dessa luta para o futuro)
    if fa not in fighter_history: fighter_history[fa] = []
    if fb not in fighter_history: fighter_history[fb] = []

    if winner == 'A':
        fighter_history[fa].append(1.0) # Vitória A
        fighter_history[fb].append(0.0) # Derrota B
    elif winner == 'B':
        fighter_history[fa].append(0.0) # Derrota A
        fighter_history[fb].append(1.0) # Vitória B
    else:
        fighter_history[fa].append(0.5) # Empate/NC
        fighter_history[fb].append(0.5)

# 6. Volta a ordenação para o padrão (Mais recente no topo)
df = df.sort_values('event_date', ascending=False).reset_index(drop=True)

# 7. Remove as colunas velhas que não vamos mais usar
colunas_remover = ['a_win_streak', 'a_loss_streak', 'b_win_streak', 'b_loss_streak']
df = df.drop(columns=colunas_remover, errors='ignore')

# 8. Salva o novo dataset
df.to_csv('ufc_dataset_v2.csv', index=False)
print("Base v2 gerada com sucesso! As colunas 'last_fight_1' a 'last_fight_5' foram criadas.")