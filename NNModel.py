import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
  def __init__(self):
    super().__init__()

# 총 Layer 수는 6개이며, 입력부터 Feature 사이즈 진행은 다음과 같다.
# ( 입력 -> 128 -> 128 -> 128 -> 64 -> 16 -> 2)
# ( 입력 -> 256 -> 128 -> 64 -> 32 -> 16 -> 2 )
# Layer1,2,3 직후에는 Batch Normalization 층이 존재한다.
#활성함수는 ReLU를 사용한다.
    self.linear1 = nn.Linear(4,128)
    self.bn1 = nn.BatchNorm1d(128)
    self.linear2 = nn.Linear(128,128)
    self.bn2 = nn.BatchNorm1d(128)
    self.linear3 = nn.Linear(128,128)
    self.bn3 = nn.BatchNorm1d(128)

    self.linear4 = nn.Linear(128,64)
    self.linear5 = nn.Linear(64,16)
    self.linear6 = nn.Linear(16,2)

  def forward(self,x):
#( Linear 층 - Batch Normalization - Activation )
    x = F.relu(self.bn1(self.linear1(x)))
    x = F.relu(self.bn2(self.linear2(x)))
    x = F.relu(self.bn3(self.linear3(x)))
    
    x = self.linear4(x)
    x = self.linear5(x)
    x = F.relu(self.linear6(x))
    return x

if __name__ == '__main__':
  pass