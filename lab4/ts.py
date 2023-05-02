import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import factor_analyzer
# Загрузка данных
df = pd.read_csv('./data.csv')

# Отбор количественных признаков
num_features = df.select_dtypes(include=[np.number]).columns.tolist()
df_num = df[num_features]

# Стандартизация данных
df_std = (df_num - df_num.mean()) / df_num.std()

# Корреляционная матрица и тепловая карта
corr_matrix = df_std.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()

# Снижение размерности с помощью PCA
pca = PCA()
pca.fit(df_std)
pca_var = pca.explained_variance_ratio_
cumsum_var = np.cumsum(pca_var)
n_components = np.argmax(cumsum_var >= 0.8) + 1
print('Количество компонент:', n_components)

pca = PCA(n_components=n_components)
pca.fit(df_std)
df_pca = pd.DataFrame(pca.transform(df_std), columns=[f'PC{i}' for i in range(1, n_components+1)])

# Диаграммы рассеяния для новой системы координат
sns.pairplot(df_pca)
plt.show()

# Факторный анализ без вращения
fa = FactorAnalyzer(rotation=None)
fa.fit(df_std, n_factors=len(df_std.columns))
print('Матрица нагрузок (без вращения):\n', fa.loadings_)
print('Матрица общностей (без вращения):\n', fa.get_communalities())

# Факторный анализ с вращением
fa = FactorAnalyzer(rotation='varimax', n_factors=2)
fa.fit(df_std, n_factors=2)
print('Матрица нагрузок (с вращением):\n', fa.loadings_)
print('Матрица общностей (с вращением):\n', fa.get_communalities())

# Сравнение результатов
sns.heatmap(fa.loadings_, cmap='coolwarm')
plt.show()
