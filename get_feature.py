import os
def dir_to_list(dir_name):
  allFileNameList = os.listdir(dir_name)
  datFileNameList = [filename for filename in allFileNameList if filename.endswith(".dat")]
  dataMatrix = []
  for datFileName in datFileNameList:
    with open(dir_name + datFileName) as file:
      lines = file.readlines()
    for line in lines:
      dataMatrix.append(line.split(','))
  return dataMatrix

import numpy as np
def get_cols(dataMatrix):
  npDataMatrix = np.array(dataMatrix)
  columns_to_keep = [3,5,40]
  data = (npDataMatrix[:, columns_to_keep])[:, [1, 0, 2]]
  data = data[data[:, 2].argsort()]
  return data

# 특징공학
import datetime 
def convert_to_minutes(datetime_str):
    datetime_obj = datetime.datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S.%f')
    year_minutes = datetime_obj.year * 365 * 24 * 60
    month_minutes = datetime_obj.month * 30 * 24 * 60
    day_minutes = datetime_obj.day * 24 * 60
    hour_minutes = datetime_obj.hour * 60
    minutes = datetime_obj.minute
    return year_minutes + month_minutes + day_minutes + hour_minutes + minutes

def convert_to_seconds(datetime_str):
    datetime_obj = datetime.datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S.%f')
    seconds = datetime_obj.second + datetime_obj.microsecond
    return seconds

# 데이터 처리 및 계산
def get_features(data):
  result = []
  for row in data:
      summary_create_time = convert_to_minutes(row[1])
      imsi = row[0]
      bearer_created_time = row[2]
      result.append([imsi, summary_create_time, bearer_created_time])

  result = np.array(result)
  unique_minutes = np.unique(result[:, 1])  # 중복 제거된 분 값들

  train_data = []
  for minute in unique_minutes:
      minute_data = result[result[:, 1] == minute]  # 해당 분에 해당하는 데이터 추출
      unique_imsi = np.unique(minute_data[:, 0])  # 해당 분에 해당하는 IMSI
      for imsi in unique_imsi:
          time_gaps = []
          imsi_data = minute_data[minute_data[:, 0] == imsi]
          bearer_created_times = imsi_data[:, 2]
          time_gaps.extend(np.diff([convert_to_seconds(time) for time in bearer_created_times]))  # BearerCreatedTime 간격 계산
          time_gaps = np.array(time_gaps)
          avg_time_gap = np.mean(time_gaps) if len(time_gaps) > 0 else 0.0
          var_time_gap = np.var(time_gaps) if len(time_gaps) > 0 else 0.0
          train_data.append([imsi, np.float64(minute), np.float64(len(bearer_created_times)), avg_time_gap, var_time_gap])
  return train_data
  # 결과 : [[imsi, 분, bearer 개수, 평균, 분산]]
def get(dir_name):
   return get_features(get_cols(dir_to_list(dir_name)))

def label_rows(data):
    labels = []
    for row in data:
        imsi = int(row[0])
        if imsi >= 450052006000001:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(labels)

if __name__ == '__main__':
   pass
