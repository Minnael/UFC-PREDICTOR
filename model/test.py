import pandas as pd

df = pd.read_csv("ufc_dataset.csv")
print(df["method"].value_counts(dropna=False))