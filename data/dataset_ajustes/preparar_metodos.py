import pandas as pd
# 1. Carrega o dataset com o histórico (o v2 que criámos anteriormente)
df = pd.read_csv('ufc_dataset_v2.csv')

# 2. Função para agrupar os métodos
def mapear_metodo(metodo):
    metodo = str(metodo).lower()
    if 'ko' in metodo: 
        return 1  # Categoria 1: Nocaute (KO/TKO)
    elif 'sub' in metodo: 
        return 2  # Categoria 2: Finalização (Submission)
    else: 
        return 0  # Categoria 0: Decisão (ou anómalos como DQ)

# 3. Cria a nova coluna
df['method_class'] = df['method'].apply(mapear_metodo)

# 4. Salva a nova versão
df.to_csv('ufc_dataset_v3.csv', index=False)
print("Base v3 gerada com sucesso! Coluna 'method_class' (0=Dec, 1=KO, 2=Sub) adicionada.")