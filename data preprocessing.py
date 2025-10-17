import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv("sensordata.csv")

# Data cleaning
data.drop_duplicates(inplace=True)
data.fillna(data.mean(), inplace=True)

# Feature extraction
data['timesincelastmaintenance'] = data['timestamp'] - data['lastmaintenance'].dt.total_seconds() / 3600

# Normalization
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[["temperature", "humidity", "vibration", "soilmoisture", "timesincelastmaintenance"]])

# Encoding categorical variables
labelencoder = LabelEncoder()
data["equipmentstatus"] = labelencoder.fit_transform(data["equipmentstatus"])
