import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from datetime import date
from keras.layers.normalization import BatchNormalization
from sklearn.preprocessing import MinMaxScaler
from pprint import pprint
from nsepy import get_history
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.models import model_from_json

#dataset = pd.read_csv('Google_Stock_Price_Train.csv',index_col="Date",parse_dates=True)

dataset = get_history(symbol='TATAMOTORS',
                   start=date(2015,9,1),
                   end=date(2019,6,12))

dataset.head() #to print the dataset heads
dataset.isna().any() #to check wheather a data column had Na values
blt = dataset['Open']
#blt.plot(figsize=(16,6), color = 'red') # To get the opening value for the entire dataset

training_set=dataset['Open']
training_set=pd.DataFrame(training_set)
# Feature scaling for training set
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)
'''
X_train = []
y_train = []
for i in range(60, len(dataset['Open'])):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Builing the Neural Network
model = Sequential()
model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50))
model.add(Dropout(0.2))
model.add(Dense(units = 1))

# Compiling the RNN
model.compile(optimizer = 'ADAM', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
model.fit(X_train, y_train, epochs =100, batch_size = 32)

# Model Saving
model_json = model.to_json()
with open("model1.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model1.h5")

'''
# Model Loading
json_file = open('model1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model1.h5")
print("Loaded model from disk")

# Getting the real stock price . If stock data is available till 2018 then the year 2018 could be your test data.
#dataset_test = pd.read_csv('Google_Stock_Price_Test.csv',index_col="Date",parse_dates=True)

dataset_test = get_history(symbol='TATAMOTORS',
                   start=date(2019,6,1),
                   end=date(2019,10,16))

real_stock_price = dataset_test.iloc[:, 4:5].values
start=date(2019,10,16)
end=date(2019,10,23)
mydates = pd.date_range(start, end).tolist()
mydates = np.array(mydates)
mydates.reshape(8,1)

mydates1 = list(dataset_test.index)
mydates1 = np.array(mydates1)
mydates1.reshape(91,1)
dataset_test.head()

#dataset_test["Volume"] = dataset_test["Volume"].str.replace(',', '').astype(float)

test_set=dataset_test['Open']
test_set=pd.DataFrame(test_set)

# Getting prediction

dataset_total = pd.concat((dataset['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
  
for i in range(60, inputs.shape[0]+1):
    X_test.append(inputs[i-60:i, 0])
X_test= np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = loaded_model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
future = []

for i in range(7):
    future.append(predicted_stock_price[-1])
    fapp = future[-1]
    #print(fapp)
    dataset_test.append({'Open' : fapp}, ignore_index = True)
    dataset_total = pd.concat((dataset['Open'], dataset_test['Open']), axis = 0)
    inputs = dataset_total[len(dataset_total) - len(dataset_test) - 67:].values
    inputs = inputs.reshape(-1,1)
    inputs = sc.transform(inputs)
    X_fut = []
    for i in range(60, inputs.shape[0]+1):
        X_fut.append(inputs[i-60:i, 0])
    X_fut= np.array(X_fut)
    X_fut = np.reshape(X_fut, (X_fut.shape[0], X_fut.shape[1], 1))
    predicted_stock_price = loaded_model.predict(X_fut)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)
    
'''
print(len(real_stock_price))
print(len(predicted_stock_price))
'''
for i in range (91,99):
        print(mydates[i-91],end = ' ' )
        print(predicted_stock_price[i])

plt.plot(mydates, predicted_stock_price[91:] ,color = 'green', label = 'Future Stock Price')
plt.plot(mydates1,real_stock_price, color = 'red', label = 'Real Stock Price')
plt.plot(mydates1,predicted_stock_price[:91], color = 'blue', label = 'Predicted Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
