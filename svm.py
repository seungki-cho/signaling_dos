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
from sklearn.svm import SVC

# SVM 모델 생성
svm = SVC(random_state=42)

# SVM 모델 학습
svm.fit(x_train, y_train)

# 테스트 데이터에 대한 예측
predictions = svm.predict(x_test)

# 예측 결과 평가
accuracy = svm.score(x_test, y_test)

print("Accuracy:", accuracy)
#0.6364245061935052