import get_feature as gf
feature_data = gf.get("./test/")
labels = gf.label_rows(feature_data)

# Train, Test 데이터 분할
from sklearn.model_selection import train_test_split
import numpy as np
output_without_imsi = np.delete(feature_data, 0, 1)
output_without_imsi = output_without_imsi.astype(np.float64)
x_train, x_test, y_train, y_test = train_test_split(output_without_imsi, labels, test_size=0.3, random_state=42)

"""### Random Forest"""
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_jobs=-1, random_state=42)
scores = cross_validate(rf, x_train, y_train, return_train_score=True, n_jobs=-1)

print(np.mean(scores['train_score']), np.mean(scores['test_score']))
# 0.9981311059480695 0.9979872662518489

rf.fit(x_train, y_train)
print(rf.feature_importances_)
# [0.12580631 0.56085631 0.14339027 0.16994711]

from sklearn.model_selection import GridSearchCV

# 매개변수 그리드 생성
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 4, 6]
}

# 랜덤 포레스트 모델 초기화
rf = RandomForestClassifier(random_state=42)

# 그리드 서치 객체 생성
grid_search = GridSearchCV(rf, param_grid, cv=5)

# 그리드 서치를 사용하여 최적의 매개변수 조합 탐색
grid_search.fit(x_train, y_train)

# 최적의 매개변수 조합 확인
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# 최적의 매개변수로 모델 생성 및 학습
best_rf = RandomForestClassifier(**best_params, random_state=42)
best_rf.fit(x_train, y_train)

# 테스트 데이터에 대한 예측
predictions = best_rf.predict(x_test)

# 예측 결과 평가
accuracy = best_rf.score(x_test, y_test)
print("Accuracy:", accuracy)

import pickle
with open('random_forest_model.pkl', 'wb') as f:
  pickle.dump(best_rf, f)