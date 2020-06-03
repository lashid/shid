# MOUNT

# Google Colaboratory
from google.colab import drive

drive.mount('/gdrive', force_remount=True)

function
ClickConnect()
{
    console.log("코랩 연결 끊김 방지");
document.querySelector("colab-toolbar-button#connect").click()
}
setInterval(ClickConnect, 60 * 1000)

# IMPORT

import os
import gc
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib
import warnings

warnings.filterwarnings(action='ignore')

root_dir = '/gdrive/My Drive/농사직설/2020 AI Friends Season 1/raw/'

from sklearn.metrics import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import lightgbm as lgb
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn import utils
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization, TimeDistributed
from tensorflow.keras.callbacks import LambdaCallback, ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam, RMSprop

# DATA OBSERVATION

root_dir = '/gdrive/My Drive/농사직설/2020 AI Friends Season 1/raw/'
csv_list = os.listdir(root_dir)
csv_list

train = pd.read_csv(root_dir + 'train.csv', encoding='utf-8')
test = pd.read_csv(root_dir + 'test.csv', encoding='utf-8')

train.tail()

temperature_name = ["X00", "X07", "X28", "X31", "X32"]  # 기온
localpress_name = ["X01", "X06", "X22", "X27", "X29"]  # 현지기압
speed_name = ["X02", "X03", "X18", "X24", "X26"]  # 풍속
water_name = ["X04", "X10", "X21", "X36", "X39"]  # 일일 누적강수량
press_name = ["X05", "X08", "X09", "X23", "X33"]  # 해면기압
sun_name = ["X11", "X14", "X16", "X19", "X34"]  # 일일 누적일사량
humidity_name = ["X12", "X20", "X30", "X37", "X38"]  # 습도
direction_name = ["X13", "X15", "X17", "X25", "X35"]  # 풍향

train.set_index('id', drop=True, inplace=True)
test.set_index('id', drop=True, inplace=True)

data_concat = pd.concat([train.loc[:, 'X00':'X39'], test.loc[:, 'X00':'X39']])
data_concat.head()

plt.figure(figsize=(20, 40))

ax1 = plt.subplot(421)
gp1_1 = sns.lineplot(data=data_concat.loc[71:4751:144, temperature_name])
gp1_2 = sns.lineplot(data=data_concat.loc[4823::144, temperature_name])
plt.title("TEMPERATURE")

ax2 = plt.subplot(422)
gp2_1 = sns.lineplot(data=data_concat.loc[71:4751:144, sun_name])
gp2_2 = sns.lineplot(data=data_concat.loc[4823::144, sun_name])
plt.title("SUN")

ax3 = plt.subplot(423)
gp3_1 = sns.lineplot(data=data_concat.loc[71:4751:144, humidity_name])
gp3_2 = sns.lineplot(data=data_concat.loc[4823::144, humidity_name])
plt.title("HUMIDITY")

ax4 = plt.subplot(424)
gp4_1 = sns.lineplot(data=data_concat.loc[71:4751:144, water_name])
gp4_2 = sns.lineplot(data=data_concat.loc[4823::144, water_name])
plt.title("RAIN")

ax5 = plt.subplot(425)
gp5_1 = sns.lineplot(data=data_concat.loc[71:4751:144, localpress_name])
gp5_2 = sns.lineplot(data=data_concat.loc[4823::144, localpress_name])
plt.title("LOCAL_PRESS")

ax6 = plt.subplot(426)
gp6_1 = sns.lineplot(data=data_concat.loc[71:4751:144, press_name])
gp6_2 = sns.lineplot(data=data_concat.loc[4823::144, press_name])
plt.title("PRESS")

ax7 = plt.subplot(427)
gp7_1 = sns.lineplot(data=data_concat.loc[71:4751:144, speed_name])
gp7_2 = sns.lineplot(data=data_concat.loc[4823::144, speed_name])
plt.title("WIND_SPEED")

ax8 = plt.subplot(428)
gp8_1 = sns.lineplot(data=data_concat.loc[71:4751:144, direction_name])
gp8_2 = sns.lineplot(data=data_concat.loc[4823::144, direction_name])
plt.title("WIND_DIRECTION")

plt.figure(figsize=(20, 40))

ax1 = plt.subplot(421)
gp1 = sns.lineplot(data=train[3888:4320].loc[:, temperature_name])
plt.title("TEMPERATURE")

ax2 = plt.subplot(422)
gp2 = sns.lineplot(data=train[3888:4320].loc[:, sun_name])
plt.title("SUN")

ax3 = plt.subplot(423)
gp3 = sns.lineplot(data=train[3888:4320].loc[:, humidity_name])
plt.title("HUMIDITY")

ax4 = plt.subplot(424)
gp4 = sns.lineplot(data=train[3888:4320].loc[:, water_name])
plt.title("RAIN")

ax5 = plt.subplot(425)
gp5 = sns.lineplot(data=train[3888:4320].loc[:, localpress_name])
plt.title("LOCAL_PRESS")

ax6 = plt.subplot(426)
gp6 = sns.lineplot(data=train[3888:4320].loc[:, press_name])
plt.title("PRESS")

ax7 = plt.subplot(427)
gp7 = sns.lineplot(data=train[3888:4320].loc[:, speed_name])
plt.title("WIND_SPEED")

ax8 = plt.subplot(428)
gp8 = sns.lineplot(data=train[:144].loc[:, direction_name])
plt.title("WIND_DIRECTION")

sensor_names1 = ['Y01', 'Y02']
sensor_names2 = ['Y03', 'Y04']
sensor_names3 = ['Y06', 'Y07']
sensor_names4 = ['Y05', 'Y12', 'Y16']
sensor_names5 = ['Y08', 'Y09', 'Y17']

control_names = ['Y18']

plt.figure(figsize=(20, 15))

ax1 = plt.subplot(321)
gp1 = sns.lineplot(data=train[3888:4320].loc[:, sensor_names1])
plt.title("sensor_names1")

ax2 = plt.subplot(322)
gp2 = sns.lineplot(data=train[3888:4320].loc[:, sensor_names2])
plt.title("sensor_names2")

ax3 = plt.subplot(323)
gp3 = sns.lineplot(data=train[3888:4320].loc[:, sensor_names3])
plt.title("sensor_names3")

ax4 = plt.subplot(324)
gp4 = sns.lineplot(data=train[3888:4320].loc[:, sensor_names4])
plt.title("sensor_names4")

ax5 = plt.subplot(325)
gp5 = sns.lineplot(data=train[3888:4320].loc[:, sensor_names5])
plt.title("sensor_names5")

ax6 = plt.subplot(326)
gp6 = sns.lineplot(data=train[4320:].loc[:, control_names])
plt.title("control_names")

# 상관계수 히트맵

train_corr = train.loc[:, "Y00":"Y18"].corr()
plt.figure(figsize=(15, 10))
ax = sns.heatmap(train_corr, cmap="RdBu", annot=True, vmin=0, vmax=1)
ax.set_ylim(len(train_corr.columns), 0)
plt.show()

# DATA PROCESSING

# 데이터 읽기

train = pd.read_csv(root_dir + 'train.csv')
test = pd.read_csv(root_dir + 'test.csv')

# 고장난 센서 제거

train.plot(x="id", y=train.columns[train.max() == train.min()])
plt.show()


def same_min_max(df):
    return df.drop(df.columns[df.max() == df.min()], axis=1)


train = same_min_max(train)
test = same_min_max(test)


# 표준화

# 정규화 함수
def standardization(df):
    mean = np.mean(df)
    std = np.std(df)
    norm = (df - mean) / (std - 1e-07)
    return norm, mean, std


X_name = train.loc[:, "X00":"X39"].columns

train_X_norm, mean, std = standardization(train[X_name])
test_X_norm = (test[X_name] - mean) / (std - 1e-07)

# 표준화된 X들과 id컬럼 병합
train2 = pd.concat([train["id"], train_X_norm], axis=1)
test2 = pd.concat([test["id"], test_X_norm], axis=1)

print(train2.shape, test2.shape)
test2.head()

# 시간 부여

minute = (train.id % 144).astype(int)
hour = pd.Series((train.index % 144 / 6).astype(int))

min_in_day = 24 * 6
hour_in_day = 24

minute_sin = np.sin(np.pi * minute / min_in_day)
minute_cos = np.cos(np.pi * minute / min_in_day)

hour_sin = np.sin(np.pi * hour / hour_in_day)
hour_cos = np.cos(np.pi * hour / hour_in_day)

t1 = range(len(minute_sin[:288]))
plt.plot(t1, minute_sin[:288],
         t1, minute_cos[:288], 'r-')
plt.title("Sin & Cos")
plt.show()

cols = list(train.columns[-19:-1])

flist = []
plist = []
nlist = []

while cols:
    col1 = cols.pop(0)
    for col2 in cols:
        lresult = stats.levene(train.loc[:4319, col1], train.loc[:4319, col2])
        flist.append(round(lresult[0], 4))
        plist.append(round(lresult[1], 4))
        nlist.append(str(col1) + '&' + str(col2))

cols = list(train.columns[-19:-1])

summary = pd.DataFrame()
summary['cols'] = nlist
summary['Fstat'] = flist
summary['FP-value'] = plist
summary['등분산'] = summary['FP-value'].apply(lambda x: '이분산' if x < 0.05 else '등분산')
summary['Tstat'] = 0
summary['TP-value'] = 0
summary['Tresult'] = 0

while cols:
    col1 = cols.pop(0)
    for col2 in cols:
        if bool(summary[summary['cols'] == str(col1) + '&' + str(col2)]['등분산'].values == '등분산'):
            ttest_result = stats.ttest_ind(train[col1], train[col2], equal_var=True)
            summary.loc[summary['cols'] == str(col1) + '&' + str(col2), 'Tstat'] = ttest_result[0]
            summary.loc[summary['cols'] == str(col1) + '&' + str(col2), 'TP-value'] = ttest_result[1]
            if round(ttest_result[1], 4) < 0.05:
                summary.loc[summary['cols'] == str(col1) + '&' + str(col2), 'Tresult'] = '차이가 있다'
            else:
                summary.loc[summary['cols'] == str(col1) + '&' + str(col2), 'Tresult'] = '차이가 없다'
        else:
            ttest_result = stats.ttest_ind(train[col1], train[col2], equal_var=False)
            summary.loc[summary['cols'] == str(col1) + '&' + str(col2), 'Tstat'] = ttest_result[0]
            summary.loc[summary['cols'] == str(col1) + '&' + str(col2), 'TP-value'] = ttest_result[1]
            if round(ttest_result[1], 4) < 0.05:
                summary.loc[summary['cols'] == str(col1) + '&' + str(col2), 'Tresult'] = '차이가 있다'
            else:
                summary.loc[summary['cols'] == str(col1) + '&' + str(col2), 'Tresult'] = '차이가 없다'

summary[lambda x: x['등분산'] == '등분산']


# Model (BASIC)

def mse_AIFrenz(y_true, y_pred):
    '''
    y_true: 실제 값
    y_pred: 예측 값
    '''
    diff = abs(y_true - y_pred)

    less_then_one = np.where(diff < 1, 0, diff)

    # multi-column일 경우에도 계산 할 수 있도록 np.average를 한번 더 씌움
    score = np.average(np.average(less_then_one ** 2, axis=0))

    return score


X = train.loc[4320:4752, 'X00':'X39']
y = train.loc[4320:4752, 'Y18']

lab_enc = preprocessing.LabelEncoder()
y = lab_enc.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=24)
print(len(X_train), len(X_test))

rf = RandomForestRegressor()
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

dt = DecisionTreeRegressor()
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

xgb = XGBRegressor()
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)

lgbm = LGBMRegressor()
lgbm.fit(X_train, y_train)
lgbm_pred = lgbm.predict(X_test)

model_names = ['rf', 'dt', 'xgb', 'lgbm', 'answer', 'rf_score', 'dt_score', 'xgb_score', 'lgbm_score']

result = pd.DataFrame()
result.colums = model_names
result['rf'] = rf_pred
result['dt'] = dt_pred
result['xgb'] = xgb_pred
result['lgbm'] = lgbm_pred
result['answer'] = y_test
result['rf_score'] = mse_AIFrenz(result['answer'], result['rf'])
result['dt_score'] = mse_AIFrenz(result['answer'], result['dt'])
result['xgb_score'] = mse_AIFrenz(result['answer'], result['xgb'])
result['lgbm_score'] = mse_AIFrenz(result['answer'], result['lgbm'])
result.head()

# MODEL (LGBM)

temperature_name = ["X00", "X07", "X28", "X31", "X32"]  # 기온
localpress_name = ["X01", "X06", "X22", "X27", "X29"]  # 현지기압
speed_name = ["X02", "X03", "X18", "X24", "X26"]  # 풍속
water_name = ["X04", "X10", "X21", "X36", "X39"]  # 일일 누적강수량
press_name = ["X05", "X08", "X09", "X23", "X33"]  # 해면기압
sun_name = ["X11", "X14", "X16", "X19", "X34"]  # 일일 누적일사량
humidity_name = ["X12", "X20", "X30", "X37", "X38"]  # 습도
direction_name = ["X13", "X15", "X17", "X25", "X35"]  # 풍향

# 데이터 읽기
root_dir = '/gdrive/My Drive/농사직설/2020 AI Friends Season 1/raw/'

train = pd.read_csv(root_dir + 'filled_train.csv')
test = pd.read_csv(root_dir + 'test.csv')

train_columns = train.columns.tolist()
test_columns = test.columns.tolist()

temp = train["Y18"].isna()
null_index = temp[temp == True].index
train.loc[null_index, "Y18"] = train.loc[null_index, "Y00":"Y17"].mean(axis=1)
dataY = train.loc[:, "Y18"]

train = pd.DataFrame(scaler.fit_transform(train), columns=train_columns)
test = pd.DataFrame(scaler.fit_transform(test), columns=test_columns)

train_drop_columns = ['X14', 'X16', 'X19'] + direction_name
test_drop_columns = ['id', 'X14', 'X16', 'X19'] + direction_name

train.drop(train_drop_columns, axis=1, inplace=True)
test.drop(test_drop_columns, axis=1, inplace=True)

dataX = train.loc[:, "X00":"X39"]
testX = test.loc[:, "X00":"X39"]

train_size = int(len(dataX) * 0.7)

X_train = np.array(dataX[0:train_size])
y_train = np.array(dataY[0:train_size])
X_test = np.array(dataX[train_size:len(dataX)])
y_test = np.array(dataY[train_size:len(dataY)])

lgb = LGBMRegressor(n_jobs=-1, random_state=42)

#######
boosting_type = ['gbdt']
n_estimators = [1000 * i for i in range(2, 6)]
learning_rate = [0.01 * i for i in range(1, 11)]
max_depth = [-1]
colsample_bytree = [1.0]
metric = ['l1', 'l2', 'rmse', 'mape', 'gamma']

parameters = {'boosting_type': boosting_type,
              'n_estimators': n_estimators,
              'learning_rate': learning_rate,
              'colsample_bytree': colsample_bytree,
              'max_depth': max_depth,
              'metric': metric}

lgb_grid = GridSearchCV(estimator=lgb, param_grid=parameters, scoring='neg_mean_squared_error', cv=5, n_jobs=-1,
                        verbose=1)

lgb_grid.fit(dataX, dataY)
lgb_pred = lgb_grid.predict(testX)
print(lgb_grid.best_params_)

submission = pd.DataFrame(lgb_pred)
submission.to_csv('/gdrive/My Drive/농사직설/2020 AI Friends Season 1/csv/lgb.csv')
submission.head()

submission.describe()

# MODEL (XGB)

temperature_name = ["X00", "X07", "X28", "X31", "X32"]  # 기온
localpress_name = ["X01", "X06", "X22", "X27", "X29"]  # 현지기압
speed_name = ["X02", "X03", "X18", "X24", "X26"]  # 풍속
water_name = ["X04", "X10", "X21", "X36", "X39"]  # 일일 누적강수량
press_name = ["X05", "X08", "X09", "X23", "X33"]  # 해면기압
sun_name = ["X11", "X14", "X16", "X19", "X34"]  # 일일 누적일사량
humidity_name = ["X12", "X20", "X30", "X37", "X38"]  # 습도
direction_name = ["X13", "X15", "X17", "X25", "X35"]  # 풍향

# 데이터 읽기
root_dir = '/gdrive/My Drive/농사직설/2020 AI Friends Season 1/raw/'

train = pd.read_csv(root_dir + 'filled_train.csv')
test = pd.read_csv(root_dir + 'test.csv')

train_columns = train.columns.tolist()
test_columns = test.columns.tolist()

temp = train["Y18"].isna()
null_index = temp[temp == True].index
train.loc[null_index, "Y18"] = train.loc[null_index, "Y00":"Y17"].mean(axis=1)
dataY = train.loc[:, "Y18"]

train = pd.DataFrame(scaler.fit_transform(train), columns=train_columns)
test = pd.DataFrame(scaler.fit_transform(test), columns=test_columns)

train_drop_columns = ['X14', 'X16', 'X19'] + direction_name
test_drop_columns = ['id', 'X14', 'X16', 'X19'] + direction_name

train.drop(train_drop_columns, axis=1, inplace=True)
test.drop(test_drop_columns, axis=1, inplace=True)

dataX = train.loc[:, "X00":"X39"]
testX = test.loc[:, "X00":"X39"]

train_size = int(len(dataX) * 0.7)

X_train = np.array(dataX[0:train_size])
y_train = np.array(dataY[0:train_size])
X_test = np.array(dataX[train_size:len(dataX)])
y_test = np.array(dataY[train_size:len(dataY)])

xgb = XGBRegressor(n_jobs=-1, random_state=42)

#####
boosting_type = ['gbdt', 'dart']
n_estimators = [int(x) for x in np.linspace(start=1000, stop=4000, num=4)]
learning_rate = [0.01 * i for i in range(1, 11)]
num_leaves = [int(i) + 1 for i in np.linspace(10, 110, num=6)]
max_bin = [int(i) for i in np.linspace(2, 12, num=6)]
min_split_gain = [0.0, 0.01]
colsample_bytree = [0.1 * i for i in range(6, 11)]
metric = ['l1', 'l2', 'rmse', 'mape', 'gamma']

parameters = {
    'learning_rate': learning_rate,
    'silent': [True],
    'n_estimators': n_estimators,
    'refit': [True],
    'metric': metric
}

xgb_grid = GridSearchCV(estimator=xgb, param_grid=parameters, scoring='neg_mean_squared_error', cv=5, n_jobs=-1,
                        verbose=1)

xgb_grid.fit(dataX, dataY)
xgb_pred = xgb_grid.predict(testX)
print(xgb_grid.best_params_)

submission = pd.DataFrame(xgb_pred)
submission.to_csv('/gdrive/My Drive/농사직설/2020 AI Friends Season 1/csv/xgb.csv')
submission.head()

submission.describe()

# MODEL (LSTM)

temperature_name = ["X00", "X07", "X28", "X31", "X32"]  # 기온
localpress_name = ["X01", "X06", "X22", "X27", "X29"]  # 현지기압
speed_name = ["X02", "X03", "X18", "X24", "X26"]  # 풍속
water_name = ["X04", "X10", "X21", "X36", "X39"]  # 일일 누적강수량
press_name = ["X05", "X08", "X09", "X23", "X33"]  # 해면기압
sun_name = ["X11", "X14", "X16", "X19", "X34"]  # 일일 누적일사량
humidity_name = ["X12", "X20", "X30", "X37", "X38"]  # 습도
direction_name = ["X13", "X15", "X17", "X25", "X35"]  # 풍향

# 데이터 읽기

train = pd.read_csv(root_dir + 'train.csv')
test = pd.read_csv(root_dir + 'test.csv')

train_columns = train.columns.tolist()
test_columns = test.columns.tolist()

temp = train["Y18"].isna()
null_index = temp[temp == True].index
train.loc[null_index, "Y18"] = train.loc[null_index, "Y00":"Y17"].mean(axis=1)
y_train = train.loc[:, "Y18"]

train = pd.DataFrame(scaler.fit_transform(train), columns=train_columns)
test = pd.DataFrame(scaler.fit_transform(test), columns=test_columns)

drop_columns = ['id', 'X14', 'X16', 'X19'] + direction_name + press_name

train.drop(drop_columns, axis=1, inplace=True)
test.drop(drop_columns, axis=1, inplace=True)

X_train = train.loc[:, "X00":"X39"]

test.head()

seq_length = 144
data_dim = len(test.columns.tolist())
output_dim = 1

dataX = []
dataY = []

for i in range(0, len(X_train) - seq_length):
    _x = np.array(X_train[i:i + seq_length])
    _y = [np.array(y_train[i + seq_length])]
    dataX.append(_x)
    dataY.append(_y)

print(dataX[0])
print(dataY[0])

# train_size=int(len(dataY)*0.7)
train_size = 4320

X_train = np.array(dataX[0:train_size])
y_train = np.array(dataY[0:train_size])
X_test = np.array(dataX[train_size:len(dataX)])
y_test = np.array(dataY[train_size:len(dataY)])

# X_train=tf.reshape(X_train, (train_size, 6, 40))
# y_train=tf.reshape(y_train, (train_size, 6, 1))
# X_test=tf.reshape(X_test, (len(dataY)-train_size, 6, 40))
# y_test=tf.reshape(y_test, (len(dataY)-train_size, 6, 1))

# X_train, X_test, y_train, y_test = train_test_split(dataX, dataY, test_size=0.3)
# X_train = np.array(X_train)
# X_test = np.array(X_test)
# y_train = np.array(y_train)
# y_test = np.array(y_test)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

print(X_train[0].shape)
print(y_train[0].shape)
print(X_train[0])
print(y_train[0])

# callback 설정
filename = 'checkpoint-epoch-{}-batch-{}-trial-001.h5'.format(EPOCHS, BATCH_SIZE)

checkpoint = ModelCheckpoint(filename,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             mode='auto')

earlystopping = EarlyStopping(monitor='val_loss',
                              patience=10)

EPOCHS = 1000
BATCH_SIZE = 128

model = Sequential()
model.add(LSTM(128, input_shape=(seq_length, data_dim)))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

opt = Adam(lr=0.001, decay=1e-6)

model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mse', 'mae', 'mape'])

model.summary()

model.fit(X_train, y_train,
          batch_size=BATCH_SIZE, epochs=EPOCHS, shuffle=False,
          validation_data=(X_test, y_test), callbacks=[checkpoint, earlystopping],
          verbose=1)

score = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE)
score

model.fit(np.array(dataX), np.array(dataY),
          batch_size=BATCH_SIZE, epochs=EPOCHS, shuffle=False)

test_list = []

for i in range(0, len(test) - seq_length):
    _x = np.array(test[i:i + seq_length])
    test_list.append(_x)
test_np = np.array(test_list)
print(test_np[0])
print(test_np.shape)

submission = model.predict(test_np)

submission = pd.DataFrame(submission)
submission.to_csv('/gdrive/My Drive/농사직설/2020 AI Friends Season 1/lstm.csv')
submission

submission.describe()