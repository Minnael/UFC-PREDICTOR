import pandas as pd
df = pd.read_csv("ufc_dataset.csv")
print(df[["a_td_per_15min", "a_sub_per_15min", "b_td_per_15min", "b_sub_per_15min"]].head(3))