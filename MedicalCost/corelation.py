import pandas as pd
from sklearn.model_selection import train_test_split

# Yeni CSV dosyasını yükleyelim
file_path = "./data/medical_cost_normalized_filtered.csv"
df = pd.read_csv(file_path)

# Veriyi kontrol edelim
print(df.head())

# Korelasyon matrisini hesaplayalım
correlation_matrix = df.corr()

# 'charges' sütunu ile olan korelasyonları alalım
charges_correlation = correlation_matrix["charges"].sort_values(ascending=False)

# 0.5 - 1 özellik, 0.3 - 2 özellik, 0.2 - 2 özellik, 0.1 - 4 özellik
threshold = 0.1  # Bu değeri istediğinize göre ayarlayabilirsiniz
selected_features = charges_correlation[abs(charges_correlation) > threshold].index

# Seçilen özellikleri kontrol edelim
print("Seçilen özellikler:")
print(selected_features)

# Seçilen özelliklere sahip yeni veri setini oluşturalım
df_selected = df[selected_features]

# Veriyi eğitim ve test setlerine ayıralım
train_df, test_df = train_test_split(df_selected, test_size=0.2, random_state=42)

# Eğitim ve test verilerini kaydedelim
train_file_path = "./updated_data/correlation_train.csv"
test_file_path = "./updated_data/correlation_test.csv"

train_df.to_csv(train_file_path, index=False)
test_df.to_csv(test_file_path, index=False)
