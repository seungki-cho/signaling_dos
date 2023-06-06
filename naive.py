from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import get_feature as gf
feature_data = gf.get("./test/")
labels = gf.label_rows(feature_data)

# Train, Test 데이터 분할
from sklearn.model_selection import train_test_split
import numpy as np
output_without_imsi = np.delete(feature_data, 0, 1)
output_without_imsi = output_without_imsi.astype(np.float64)
x_train, x_test, y_train, y_test = train_test_split(output_without_imsi, labels, test_size=0.3, random_state=42)

# Naive Bayes 모델 초기화
nb = GaussianNB()

# 모델 학습
nb.fit(x_train, y_train)

# 테스트 데이터에 대한 예측
predictions = nb.predict(x_test)

# 정확도 평가
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)