# 상관 관계 분석 (선형)과 랜덤포레스트 (비선형) 를 통해 result 값과 유의미한 값을 가진 변수를 추려냄
# 상관 관계 분석을 통해서는 통계적인 유의미함을 보여주기 힘들다 > 비선형성 분석 사용

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_ind
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 데이터 로드 및 전처리
file_path = 'data2.csv'
data = pd.read_csv(file_path)

# 데이터 정규화 및 필요한 전처리
data_clean = data.drop(columns=['date']).fillna(data.drop(columns=['date']).mean())
X = data_clean.drop(columns=['result'])
y = data_clean['result']

# 데이터 정규화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. 유의미한 변수 선택
# 상관 관계 분석 추가
correlation_matrix = data_clean.corr()
result_corr = correlation_matrix['result'].sort_values(ascending=False)

# 상관 관계 히트맵 시각화
plt.figure(figsize=(20, 20))
sns.heatmap(correlation_matrix, fmt=".2f", cmap='coolwarm', annot_kws={"size": 7}, cbar_kws={'shrink': .5})
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.title("Correlation Matrix")
plt.show()

# 2-1. 랜덤 포레스트 중요도를 통해 변수 선택
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_scaled, y)
importances = rf.feature_importances_
indices = importances.argsort()[::-1]
top_features_rf = X.columns[indices]

# 상위 10개의 중요 변수 출력 및 시각화
top_10 = 10
top_features_rf_10 = top_features_rf[:top_10]
top_importances_10 = importances[indices][:top_10]

plt.figure(figsize=(10, 6))
plt.title("Top 10 Feature Importances")
plt.barh(range(top_10), top_importances_10[::-1], align='center')
plt.yticks(range(top_10), top_features_rf_10[::-1])
plt.xlabel("Feature Importance")
plt.show()

# 상위 6개의 중요 변수 선택 > 유의미하게 데이터가 변화하는 구간
top_6 = 6
top_features_rf_6 = top_features_rf[:top_6]

# 3. 통계적 유의미성 판단
significant_features = []
p_values = []  # p-값을 저장할 리스트
for feature in top_features_rf_6:
    group1 = X[feature][y == 1]
    group2 = X[feature][y != 1]
    t_stat, p_value = ttest_ind(group1, group2)
    print(f"{feature}: t-statistic = {t_stat:.5f}, p-value = {p_value:.5f}")
    if p_value < 0.05:  # 유의수준 5% 기준
        significant_features.append(feature)
        p_values.append(p_value)  # 피처와 p-값을 저장

# 선택한 변수들로 데이터 구성
X_selected = data_clean[significant_features]

# 선택한 변수들에 대해서 다시 정규화
X_selected_scaled = scaler.fit_transform(X_selected)

# 4. 분류 알고리즘 비교
models = {
    "Logistic Regression": LogisticRegression(random_state=42, multi_class='multinomial'),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(probability=True, random_state=42)
}

# 평가 지표
scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score, average='macro', zero_division=0),
    'recall': make_scorer(recall_score, average='macro', zero_division=0),
    'f1': make_scorer(f1_score, average='macro', zero_division=0)
}

results = {metric: {} for metric in scoring.keys()}

for metric, score_func in scoring.items():
    for model_name, model in models.items():
        scores = cross_validate(model, X_selected_scaled, y, cv=5, scoring={metric: score_func})
        results[metric][model_name] = {
            f"mean_{metric}": scores[f'test_{metric}'].mean(),
            f"std_{metric}": scores[f'test_{metric}'].std()
        }

# 모델 성능 출력
metrics = ['accuracy', 'precision', 'recall', 'f1']

for metric in metrics:
    print(f"{metric.capitalize()} scores:")
    for model in models:
        mean_score = results[metric][model][f"mean_{metric}"]
        std_score = results[metric][model][f"std_{metric}"]
        print(f"{model}: {mean_score:.4f} ± {std_score:.4f}")

# 모델 성능 비교 시각화
metrics = ['accuracy', 'precision', 'recall', 'f1']

for metric in metrics:
    fig, ax = plt.subplots(figsize=(10, 6))
    values = [results[metric][model][f"mean_{metric}"] for model in models]
    stds = [results[metric][model][f"std_{metric}"] for model in models]
    ax.bar(models.keys(), values, capsize=5)
    ax.set_title(f"Model {metric.capitalize()}")
    ax.set_ylim(0, 1)
    plt.show()

# 6. 특정 조건에서의 예측
specific_condition = pd.DataFrame(columns=significant_features)
specific_condition.loc[0] = [0] * len(significant_features)

# 특정 조건 설정
if 'minutes' in significant_features:
    specific_condition['minutes'] = 40
if 'goals' in significant_features:
    specific_condition['goals'] = 0
if 'shots' in significant_features:
    specific_condition['shots'] = 0
if 'shots_on_target' in significant_features:
    specific_condition['shots_on_target'] = 0
if 'venue' in significant_features: # 인코딩 과정으로 인해 0 > home, 1 > 어웨이, 2 > 중립국
    specific_condition['venue'] = 1

# 피처 순서 맞추기
specific_condition = specific_condition[X_selected.columns]

# 정규화
specific_condition_scaled = scaler.transform(specific_condition)

# 각 모델에 대한 예측
predictions = {}
for model_name, model in models.items():
    model.fit(X_selected_scaled, y)
    predicted_result_proba = model.predict_proba(specific_condition_scaled)
    predicted_result = np.argmax(predicted_result_proba) - 1  # -1, 0, 1 중 하나로 변환
    predictions[model_name] = {
        "predicted_result": predicted_result,
        "probabilities": predicted_result_proba[0]
    }

    # 결과 값이 -1 > 패배, 0 > 무승부, 1 > 승리
for model_name, prediction in predictions.items():
    print(f"Predicted Result ({model_name}): {prediction['predicted_result']}")
    print(f"Prediction Probabilities ({model_name}): {prediction['probabilities']}")
    print("-" * 30)
