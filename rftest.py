test_data_dir = "./ttest/"

import get_feature as gf

import pickle
with open('random_forest_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

import numpy as np
data = gf.get(test_data_dir)
data_without_imsi = np.delete(data, 0, 1)
removed_imsi = [row[0] for row in data]

predictions = loaded_model.predict(data_without_imsi)

data_with_predictions = np.column_stack((removed_imsi, data_without_imsi, predictions))

print("기존 데이터 수 : ", len(gf.dir_to_list(test_data_dir)))
print("특징 추출 후 데이터 수 : ", len(data_without_imsi))

result = {}
# imsi 별로 개수를 측정
for row in data_with_predictions:
    imsi = row[0]
    count = int(float(row[2]))
    isAttack = int(row[5])
    if isAttack == 1:
        if imsi in result:
            result[imsi] += count
        else:
            result[imsi] = count

# imsi 별로 정렬
sorted_result = sorted(result.items(), key=lambda x: x[0])

# 결과를 csv 파일로 저장
import csv
filename = 'result.csv'
with open(filename, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['IMSI', '공격수'])  # 헤더
    writer.writerows(sorted_result)