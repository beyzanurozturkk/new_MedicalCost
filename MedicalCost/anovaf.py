import pandas as pd
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.model_selection import train_test_split

# Yeni CSV dosyasını yükleyelim
file_path = "./data/medical_cost_normalized_filtered.csv"
df = pd.read_csv(file_path)

# Veriyi kontrol edelim
print(df.head())

# 'charges' sütununu hedef değişken olarak belirleyelim
X = df.drop("charges", axis=1)
y = df["charges"]

# ANOVA F-test yöntemini kullanarak en iyi özellikleri seçelim
selector = SelectKBest(score_func=f_classif, k="all")
selector.fit(X, y)

# Seçilen özelliklerin skorlarını alalım
scores = selector.scores_
features_scores = pd.DataFrame({"Feature": X.columns, "Score": scores})
features_scores = features_scores.sort_values(by="Score", ascending=False)

# Belirli bir eşik değere göre en iyi özellikleri seçelim
threshold = 0.01  # Bu değeri istediğinize göre ayarlayabilirsiniz
selected_features = features_scores[features_scores["Score"] > threshold][
    "Feature"
].values

# Seçilen özellikleri kontrol edelim
print("Seçilen özellikler:")
print(selected_features)

# Seçilen özelliklere sahip yeni veri setini oluşturalım
df_selected = df[selected_features]
df_selected["charges"] = y

# Veriyi eğitim ve test setlerine ayıralım
train_df, test_df = train_test_split(df_selected, test_size=0.2, random_state=42)

# Eğitim ve test verilerini kaydedelim
train_file_path = "./updated_data/anova_correlation_train.csv"
test_file_path = "./updated_data/anova_correlation_test.csv"

train_df.to_csv(train_file_path, index=False)
test_df.to_csv(test_file_path, index=False)
