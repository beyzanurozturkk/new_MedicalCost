import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE

# Yeni CSV dosyasını yükleyelim
file_path = "./data/medical_cost_normalized_filtered.csv"
df = pd.read_csv(file_path)

# Veriyi kontrol edelim
print(df.head())

# 'charges' sütununu hedef değişken olarak belirleyelim
X = df.drop("charges", axis=1)
y = df["charges"]

# Veriyi standartlaştırmak için scaler kullanalım
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Lasso modelini tanımlayalım
lasso = Lasso(alpha=10)

# RFE kullanarak özellik seçimi yapalım
selector = RFE(lasso, n_features_to_select=5, step=1)
selector = selector.fit(X_scaled, y)

# Seçilen özellikleri alalım
selected_features = X.columns[selector.support_]

# Seçilen özellikleri kontrol edelim
print("Seçilen özellikler:")
print(selected_features)

# Seçilen özelliklere sahip yeni veri setini oluşturalım
df_selected = df[selected_features]
df_selected["charges"] = y

# Veriyi eğitim ve test setlerine ayıralım
train_df, test_df = train_test_split(df_selected, test_size=0.2, random_state=42)

# Eğitim ve test verilerini kaydedelim
train_file_path = "./updated_data/rfe_lasso_train.csv"
test_file_path = "./updated_data/rfe_lasso_test.csv"

train_df.to_csv(train_file_path, index=False)
test_df.to_csv(test_file_path, index=False)

print(f"Eğitim verisi {train_file_path} konumuna kaydedildi.")
print(f"Test verisi {test_file_path} konumuna kaydedildi.")
