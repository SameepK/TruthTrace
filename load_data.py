import pandas as pd


df_fake = pd.read_csv("fake.csv")
df_real = pd.read_csv("true.csv")

df_fake['label'] = "FAKE"
df_real['label'] = "REAL"

#combine datasets
df = pd.concat([df_fake, df_real]).sample(frac = 1).reset_index(drop=True)

print("Dataset size:", len(df))
print("Example row:")
print(df[['title', 'label']].head(1))
