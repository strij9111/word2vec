# Используется датасет из домашнего задания №1 - судебные дела одного из участков мировых судей
# Пытаемся по названию истца установить категорию дела

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV
from sentence_transformers import SentenceTransformer

df = pd.read_csv('court_cases.csv')
train, test = train_test_split(df, test_size=0.3, random_state=2201)

vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), min_df=5, max_df=0.8)
X_train = vectorizer.fit_transform(train['Истец'])
y_train = train['Категория']
X_test = vectorizer.transform(test['Истец'])
y_test = test['Категория']

print(y_test)

params = {
    'max_depth': [10, 20, 30],
    'learning_rate': [0.001, 0.01, 0.1, 1],
    'n_estimators': [10, 20, 50, 100, 200, 300]
}

clf = LGBMClassifier(random_state=2201)
grid_search = GridSearchCV(clf, params, cv=3, n_jobs=-1)
grid_search.fit(X_train, y_train)

print(f'Best params: {grid_search.best_params_}')
print(f'Train score: {grid_search.best_score_}')
print(f'Test score: {grid_search.score(X_test, y_test)}')

model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', device='cpu')

# векторизация обучающих данных
X_train_word2vec = []
for sentence in train['Истец']:
    X_train_word2vec.append(model.encode(sentence))

print(len(X_train_word2vec[0]))

# векторизация тестовых данных
X_test_word2vec = []
for sentence in test['Истец']:
    X_test_word2vec.append(model.encode(sentence))

clf = LGBMClassifier(random_state=2201)
grid_search = GridSearchCV(clf, params, cv=3, n_jobs=-1)
grid_search.fit(X_train_word2vec, y_train)

print(f'Best params bert: {grid_search.best_params_}')
print(f'Train score bert: {grid_search.best_score_}')
print(f'Test score bert: {grid_search.score(X_test_word2vec, y_test)}')
