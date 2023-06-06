import get_feature as gf
feature_data = gf.get("./test/")
labels = gf.label_rows(feature_data)

# Train, Test 데이터 분할
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import numpy as np
output_without_imsi = np.delete(feature_data, 0, 1)
output_without_imsi = output_without_imsi.astype(np.float64)
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(output_without_imsi)
x_train, x_test, y_train, y_test = train_test_split(normalized_data, labels, test_size=0.3, random_state=42)


# Numpy 에서 Tensor로 타입캐스팅
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


x_train = torch.from_numpy(x_train).float()
y_train = torch.from_numpy(y_train).long()

x_test = torch.from_numpy(x_test).float()
y_test = torch.from_numpy(y_test).long()

# Generating dataset
train_set = TensorDataset(x_train, y_train)
test_set = TensorDataset(x_test, y_test)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=16, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=16, shuffle=True)
# 모델 구성
import torch.nn.functional as F
import NNModel

model = NNModel.Model()
print(model)

# Configure optimizer
import torch.optim as optim

# 학습률 0.001
# Momentum 0.9

optimizer = optim.SGD(model.parameters(), lr=0.001,momentum =0.9)

# 손실함수 생성
criterion = nn.CrossEntropyLoss()

# Training

epochs = 400
losses = list()
accuracies = list()

for epoch in range(epochs):
  epoch_loss = 0  
  epoch_accuracy = 0


  for x, y in train_loader:
    x_train, y_train = x, y

    # 변화도(Gradient) 매개변수를 0으로 만들고
    optimizer.zero_grad()

    # 순전파 + 역전파 + 최적화를 한 후
    output = model(x_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()

    _, predicted = torch.max(output, dim=1)
    accuracy = (predicted == y).sum().item()
    epoch_loss += loss.item()
    epoch_accuracy += accuracy
  

  epoch_loss /= len(train_loader)
  epoch_accuracy /= len(x_train)
  print("epoch :{}, \tloss :{}, \taccuracy :{}".format(str(epoch+1).zfill(3),round(epoch_loss,4), round(epoch_accuracy,4)))
  
  losses.append(epoch_loss)
  accuracies.append(epoch_accuracy)

  model.eval()  # 모델을 평가 모드로 변경
  test_loss = 0
  test_accuracy = 0
  total_samples = 0

  with torch.no_grad():  # 그래디언트 계산 비활성화
      for x_test, y_test in test_loader:
          output = model(x_test)
          loss = criterion(output, y_test)
          _, predicted = torch.max(output, dim=1)
          accuracy = (predicted == y_test).sum().item()

          test_loss += loss.item() * x_test.size(0)
          test_accuracy += accuracy
          total_samples += x_test.size(0)

  test_loss /= total_samples
  test_accuracy /= total_samples
  print("Test Loss: {:.4f}, Test Accuracy: {:.4f}".format(test_loss, test_accuracy))


# Plot result

import matplotlib.pyplot as plt

plt.figure(figsize=(20,5))
plt.subplots_adjust(wspace=0.2)

plt.subplot(1,2,1)
plt.title("$loss$",fontsize = 18)
plt.plot(losses)
plt.grid()
plt.xlabel("$epochs$", fontsize = 16)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)

plt.subplot(1,2,2)
plt.title("$accuracy$", fontsize = 18)
plt.plot(accuracies)
plt.grid()
plt.xlabel("$epochs$", fontsize = 16)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)

plt.show()

torch.save(model.state_dict(), 'nn1.pt')