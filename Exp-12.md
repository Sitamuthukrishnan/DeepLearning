Code:
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
# Load dataset (example: one-column CSV of stock prices)
data = pd.read_csv('data.csv')
values = data['Price'].values.reshape(-1, 1)
# Normalize data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(values)
# Create sequences
def create_dataset(data, step=5):
X, y = [], []
for i in range(len(data) - step):
X.append(data[i:i+step])
y.append(data[i+step])
return np.array(X), np.array(y)
X, y = create_dataset(scaled_data, 10)
# Train-test split
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
# Build LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=False, input_shape=(X_train.shape[1], 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# Train model
model.fit(X_train, y_train, epochs=20, batch_size=16)
# Predict
predicted = model.predict(X_test)
predicted = scaler.inverse_transform(predicted)
y_test_actual = scaler.inverse_transform(y_test)
# Plot
plt.plot(y_test_actual, label='Actual')
plt.plot(predicted, label='Predicted')
plt.title('LSTM Time-Series Forecasting')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()
