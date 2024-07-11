import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import pandas as pd

# 폰트 설정
plt.rcParams['font.family'] = 'Arial'

# 데이터 로드 및 전처리
file_path = 'data2.csv'
data = pd.read_csv(file_path)

# 필요한 변수만 선택
data_clean = data[['minutes', 'goal', 'shots', 'shots_on_target', 'venue', 'result']].fillna(data.mean(numeric_only=True))

X = data_clean[['minutes', 'goal', 'shots', 'shots_on_target', 'venue']]
y = data_clean['result']

# 데이터 정규화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 훈련 데이터와 테스트 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# SVM 모델 훈련
svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(X_train, y_train)

# 테스트 데이터에 대한 예측 및 성능 평가
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"SVM Model Accuracy: {accuracy:.4f}")
print(f"SVM Model Precision: {precision:.4f}")
print(f"SVM Model Recall: {recall:.4f}")
print(f"SVM Model F1 Score: {f1:.4f}")

# 분류 리포트 출력
target_names = ['Defeat', 'Draw', 'Victory']
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))

# 하이퍼파라미터 그리드 정의
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf']
}

# 그리드 서치 수행
grid_search = GridSearchCV(SVC(), param_grid, refit=True, verbose=2, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# 최적의 하이퍼파라미터 및 성능 출력
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Score: {grid_search.best_score_:.4f}")

# 최적의 모델로 테스트 데이터 예측 및 평가
best_svm_model = grid_search.best_estimator_
y_pred_best = best_svm_model.predict(X_test)

accuracy_best = accuracy_score(y_test, y_pred_best)
precision_best = precision_score(y_test, y_pred_best, average='weighted', zero_division=0)
recall_best = recall_score(y_test, y_pred_best, average='weighted')
f1_best = f1_score(y_test, y_pred_best, average='weighted')

print(f"Tuned SVM Model Accuracy: {accuracy_best:.4f}")
print(f"Tuned SVM Model Precision: {precision_best:.4f}")
print(f"Tuned SVM Model Recall: {recall_best:.4f}")
print(f"Tuned SVM Model F1 Score: {f1_best:.4f}")

# 분류 리포트 출력
print("Tuned Classification Report:")
print(classification_report(y_test, y_pred_best, target_names=target_names, zero_division=0))

# 예측 함수
def predict_result(minutes, goals, shots, shots_on_target, venue, model, scaler):
    input_data = pd.DataFrame([[minutes, goals, shots, shots_on_target, venue]],
                              columns=['minutes', 'goal', 'shots', 'shots_on_target', 'venue'])
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)
    return int(prediction[0])

# 예측 예시
minutes = 90
goals = 0
shots = 2
shots_on_target = 1
venue = 1

result = predict_result(minutes, goals, shots, shots_on_target, venue, best_svm_model, scaler)
print("Predicted Result:", target_names[result])

# 혼동 행렬
conf_matrix = confusion_matrix(y_test, y_pred_best)

# 혼동 행렬 시각화
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# 분류 리포트를 데이터프레임으로 변환하여 시각화
report = classification_report(y_test, y_pred_best, target_names=target_names, output_dict=True)
report_df = pd.DataFrame(report).transpose()

# 분류 리포트 시각화
plt.figure(figsize=(10, 6))
sns.heatmap(report_df.iloc[:-1, :].T, annot=True, cmap='viridis')
plt.title('Classification Report')
plt.show()
