import torch
import NNModel

model = NNModel.Model()
model.load_state_dict(torch.load('nn.pt'))

import get_feature as gf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


data = gf.get("./ttest/")
removeimsi = np.delete(data, 0, 1)
float_data = removeimsi.astype(np.float64)
normalized_data = scaler.fit_transform(float_data)
test_torch = torch.from_numpy(normalized_data).float()
output = model(test_torch)

predicted_labels = torch.argmax(output, dim=1)  # 예측값 중 가장 큰 값의 인덱스를 가져옴
num_predicted_ones = torch.sum(predicted_labels == 1).item()

print(num_predicted_ones, "/" ,len(predicted_labels))
print(np.count_nonzero(gf.label_rows(data) == 1), "/", len(data))