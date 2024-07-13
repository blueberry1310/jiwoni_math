import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

font_path = r'C:\Windows\Fonts\malgun.ttf'
font_prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()

def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    X = data.drop('체육관 장비 상태 만족도', axis=1)
    y = data['체육관 장비 상태 만족도']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

X_train, X_test, y_train, y_test = load_and_preprocess_data('../data/processed_survey_data.csv')

activations = ['relu', 'logistic', 'tanh', 'identity']
results = {activation: {} for activation in activations}

for activation in activations:
    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation=activation,
        alpha=0.0001,
        solver='adam',
        max_iter=1000,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    results[activation]['accuracy'] = accuracy_score(y_test, y_pred)
    results[activation]['precision'] = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    results[activation]['recall'] = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    results[activation]['f1_score'] = f1_score(y_test, y_pred, average='weighted', zero_division=0)

results_df = pd.DataFrame(results).T
print(results_df)

best_activation = results_df['f1_score'].idxmax()
print(f"\n최고 성능의 활성화 함수: {best_activation}")

# 시각화
metrics = ['accuracy', 'precision', 'recall', 'f1_score']
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('활성화 함수별 성능 비교', fontproperties=font_prop)

for i, metric in enumerate(metrics):
    ax = axes[i // 2, i % 2]
    results_df[metric].plot(kind='bar', ax=ax)
    ax.set_title(metric, fontproperties=font_prop)
    ax.set_ylim(0, 1)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.show()