# -*- coding: utf-8 -*-
"""

@author: zorikhin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import StandardScaler
import seaborn as sns

df_production = pd.read_csv('COMPANY_BASIN_20200128 Producing Entity Monthly Production.CSV')

df_production_group = df_production.groupby('Entity ID')['API/UWI'].agg(['count'])
df_production_group.sort_values('count', ascending=False, inplace=True)

well_1 = df_production_group.index[0]

well_1_prod = df_production[df_production['Entity ID'] == well_1]

plt.plot(well_1_prod['Monthly Production Date'], well_1_prod['Monthly Oil'])

train_dates = pd.to_datetime(well_1_prod['Monthly Production Date'])
train_values = well_1_prod['Monthly Oil'].astype(float).values.reshape(-1, 1)

scaler = StandardScaler()
scaler = scaler.fit(train_values)
train_values_scaled = scaler.transform(train_values)
train_values_scaled_t = train_values_scaled[2:]

plt.plot(well_1_prod['Monthly Production Date'].values[2:], train_values_scaled_t)

trainX = []
trainY = []

n_future = 1
n_past = 12

for i in range(n_past, len(train_values_scaled_t) - n_future +1):
    trainX.append(train_values_scaled_t[i - n_past:i])
    trainY.append(train_values_scaled_t[i + n_future - 1:i + n_future])
    
trainX, trainY = np.array(trainX), np.array(trainY)

print('trainX shape == {}.'.format(trainX.shape))
print('trainY shape == {}.'.format(trainY.shape))

model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
model.add(LSTM(32, activation='relu', return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(trainY.shape[1]))

model.compile(optimizer='adam', loss='mse')
model.summary()

history = model.fit(trainX, trainY, epochs=5, batch_size=16, validation_split=0.1, verbose=1)

plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()

prediction = model.predict(trainX[-10:])
y_pred_future = scaler.inverse_transform(prediction)

df_forecast = pd.DataFrame({'Date':train_dates.iloc[-10:].values, 'Production':y_pred_future.reshape(1, -1).flatten()})
df_forecast['Date']=pd.to_datetime(df_forecast['Date'])

df_original = pd.DataFrame({'Date':train_dates.iloc[-10:].values, 'Production':train_values[-10:].reshape(1, -1).flatten()})
df_original['Date']=pd.to_datetime(df_forecast['Date'])

sns.lineplot(df_original['Date'], df_original['Production'], color='blue')
sns.lineplot(df_forecast['Date'], df_forecast['Production'], color='red')