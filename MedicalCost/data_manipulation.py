import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy import stats

# CSV dosyasını yükleyelim
file_path = "./data/medical_cost.csv"
df = pd.read_csv(file_path)

# Veriyi kontrol edelim
print(df.head())

# 'sex' sütununu sayısal değerlere çevirelim
df["sex"] = df["sex"].map({"male": 0, "female": 1})

# 'smoker' sütununu sayısal değerlere çevirelim
df["smoker"] = df["smoker"].map({"yes": 1, "no": 0})

# 'region' sütununu one-hot encoding ile sayısal değerlere çevirelim
df = pd.get_dummies(df, columns=["region"])

# 'charges' sütununu ayrı bir değişkende saklayalım
charges = df["charges"]
df = df.drop(columns=["charges"])

# Normalizasyon için scaler nesnesini oluşturalım
scaler = MinMaxScaler()

# Tüm sütunları normalleştirelim
df[df.columns] = scaler.fit_transform(df[df.columns])

# 'charges' sütununu en sona ekleyelim
df["charges"] = charges

# Dönüştürülmüş veriyi kontrol edelim
print(df.head())

# NaN değerleri kontrol et
print("NaN values in dataframe:")
print(df.isna().sum())

# NaN değer yok, devam edilebilir
# Z-score yöntemi ile outlier'ları çıkaralım
z_scores = stats.zscore(df)
abs_z_scores = abs(z_scores)
filtered_entries = (abs_z_scores < 3).all(axis=1)
df_filtered = df[filtered_entries]

# Dönüştürülmüş ve outlier'ları çıkarılmış veriyi kaydedelim
output_file_path_filtered = "./data/medical_cost_normalized_filtered.csv"
df_filtered.to_csv(output_file_path_filtered, index=False)
