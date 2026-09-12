import pandas as pd

def split_datasets_by_gender(input_csv="ufc_dataset.csv", out_male="ufc_dataset_male_only.csv", out_female="ufc_dataset_female_only.csv"):
    df = pd.read_csv(input_csv)
    
    # 1. Semente: No UFC, APENAS mulheres lutam na categoria Peso-Palha (até ~53kg)
    seed_fights = df[(df['a_weight_kg'] <= 53.0) & (df['b_weight_kg'] <= 53.0)]
    female_fighters = set(seed_fights['fighter_a']).union(set(seed_fights['fighter_b']))
    
    # 2. Semente de pesos maiores para garantir cobertura total
    famous_females = [
        "Amanda Nunes", "Ronda Rousey", "Miesha Tate", "Cris Cyborg", 
        "Holly Holm", "Julianna Pena", "Raquel Pennington", "Valentina Shevchenko",
        "Germaine de Randamie", "Megan Anderson", "Norma Dumont", "Ketlen Vieira",
        "Irene Aldana", "Yana Santos", "Macy Chiasson", "Felicia Spencer"
    ]
    female_fighters.update(famous_females)
    
    # 3. Propagação de Rede: Quem lutou contra uma mulher, também é mulher.
    while True:
        start_len = len(female_fighters)
        
        mask_a = df['fighter_a'].isin(female_fighters)
        mask_b = df['fighter_b'].isin(female_fighters)
        
        new_a = set(df[mask_b]['fighter_a'])
        new_b = set(df[mask_a]['fighter_b'])
        
        female_fighters.update(new_a)
        female_fighters.update(new_b)
        
        if len(female_fighters) == start_len:
            break
            
    print(f"Total de lutadoras mapeadas: {len(female_fighters)}")
    
    # 4. Criar máscara e dividir
    is_female_fight = df['fighter_a'].isin(female_fighters) | df['fighter_b'].isin(female_fighters)
    
    df_female_only = df[is_female_fight].copy()
    df_male_only = df[~is_female_fight].copy()
    
    print(f"Lutas femininas extraídas: {len(df_female_only)}")
    print(f"Lutas masculinas extraídas: {len(df_male_only)}")
    
    # 5. Salvar novos datasets
    df_female_only.to_csv(out_female, index=False)
    df_male_only.to_csv(out_male, index=False)
    
    print(f"\n✅ Datasets salvos:\n -> {out_male}\n -> {out_female}")

if __name__ == "__main__":
    split_datasets_by_gender()