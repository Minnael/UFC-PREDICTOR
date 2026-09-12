import pandas as pd

def get_weight_class(weight_kg):
    if pd.isna(weight_kg): return 'Unknown'
    if weight_kg <= 53.0: return 'Strawweight (52kg)'
    if weight_kg <= 58.0: return 'Flyweight (57kg)'
    if weight_kg <= 62.0: return 'Bantamweight (61kg)'
    if weight_kg <= 67.0: return 'Featherweight (65kg)'
    if weight_kg <= 71.0: return 'Lightweight (70kg)'
    if weight_kg <= 78.0: return 'Welterweight (77kg)'
    if weight_kg <= 85.0: return 'Middleweight (84kg)'
    if weight_kg <= 94.0: return 'Light Heavyweight (93kg)'
    return 'Heavyweight (120kg)'

def is_under_x_5(row, target_round):
    r = row['round']
    t = str(row['time'])
    try:
        r = int(r)
        if r < target_round: return True
        if r > target_round: return False
        mins, secs = map(int, t.split(':'))
        total_sec = mins * 60 + secs
        return total_sec <= 150
    except:
        return False

def main():
    print("Analisando o dataset histórico para Fatores de Desconto de Tempo...")
    df = pd.read_csv('global_distribuido/ufc_dataset.csv')
    
    finishes = df[df['method'].isin(['KO/TKO', 'SUB'])].copy()
    finishes['weight_class'] = finishes['a_weight_kg'].apply(get_weight_class)
    
    finishes['under_15'] = finishes.apply(lambda row: is_under_x_5(row, 2), axis=1)
    finishes['under_25'] = finishes.apply(lambda row: is_under_x_5(row, 3), axis=1)
    
    classes_order = [
        'Strawweight (52kg)', 'Flyweight (57kg)', 'Bantamweight (61kg)', 
        'Featherweight (65kg)', 'Lightweight (70kg)', 'Welterweight (77kg)', 
        'Middleweight (84kg)', 'Light Heavyweight (93kg)', 'Heavyweight (120kg)'
    ]
    
    table_lines = []
    table_lines.append("| Categoria | Total Via Rápida | Acabam < 2.5 Rounds | Fator Under 2.5 | Acabam < 1.5 Rounds | Fator Under 1.5 |")
    table_lines.append("|---|---|---|---|---|---|")
    
    for wc in classes_order:
        subset = finishes[finishes['weight_class'] == wc]
        if len(subset) == 0: continue
        
        total_finishes = len(subset)
        u25_count = subset['under_25'].sum()
        u15_count = subset['under_15'].sum()
        
        u25_pct = (u25_count / total_finishes) * 100
        u15_pct = (u15_count / total_finishes) * 100
        
        line = f"| {wc} | {total_finishes} | {u25_count} ({u25_pct:.1f}%) | {u25_pct/100:.2f} | {u15_count} ({u15_pct:.1f}%) | {u15_pct/100:.2f} |"
        table_lines.append(line)
        
    markdown_content = "\n".join(table_lines)
    
    with open('time_discount_factors.md', 'w', encoding='utf-8') as f:
        f.write("# Fatores de Desconto de Tempo por Categoria (EV+ Betting)\n\n")
        f.write("Esta tabela mostra a probabilidade histórica de uma luta que NÃO chega à decisão terminar antes das linhas de 2.5 e 1.5 rounds.\n")
        f.write("Use a coluna 'Fator' multiplicada pela chance de Via Rápida do PDF para achar a Odd Justa.\n\n")
        f.write(markdown_content)
        
    print("Tabela gerada com sucesso em: scripts/time_discount_factors.md")
    print(markdown_content)

if __name__ == '__main__':
    main()
