import get_feature as gf
feature_data = gf.get("./test/")
labels = gf.label_rows(feature_data)

# Train, Test 데이터 분할
from sklearn.model_selection import train_test_split
import numpy as np

x_train, x_test, y_train, y_test = train_test_split(feature_data, labels, test_size=0.3, random_state=42)

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

import random

random_data = feature_data
random_values = [random.randint(450051001000001, 450052009000001) for _ in range(len(random_data))]

for i in range(len(random_data)):
    random_data[i][0] = random_values[i]

import numpy as np
random_prediction = rf.predict(random_data)

def label_rows(data):
    labels = []
    for row in data:
        imsi = int(row[0])
        if imsi >= 450052006000001:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(labels)

print(np.count_nonzero(random_prediction==1), len(random_prediction))
print(np.count_nonzero(labels==1), len(labels))