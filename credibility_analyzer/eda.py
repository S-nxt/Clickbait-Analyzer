import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv("data/Fake_Real_News_Data.csv")
df['full_text'] = df['title'].fillna('') + " " + df['text'].fillna('')
df['word_count'] = df['full_text'].apply(lambda x: len(x.split()))

plt.figure(figsize=(10, 5))
sns.histplot(data=df, x='word_count', hue='label', element="step", stat="density", common_norm=False, bins=50, kde=True)
plt.xlim(0, 3000)
plt.title("Article Word Count Distribution: Real vs Fake")
plt.xlabel("Word Count")
plt.savefig("eda_word_count_distribution.png")
plt.show()
# df = pd.read_csv("data/Fake_Real_News_Data.csv")
# print(df.head())
# print(df.shape())
# print(df.isnull().sum())
# print(df.duplicated().sum())
# print(df.columns)