import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
import seaborn as sns

st.title("Reliance Industries Stock Data Application")

data = pd.read_csv('C:\\Users\\Lenovo-PC\\Downloads\\Reliance data (1).csv')
data['Date'] = pd.to_datetime(data['Date'])  
data.set_index('Date', inplace=True) 
st.subheader('Raw Data')   
st.dataframe(data)

st.header('Data Summary')
st.write(data.describe())
 
if st.button('Visualizations'):
 st.subheader("Close Price Over Time")
 fig, ax = plt.subplots()
 ax.plot(data.index, data['Close '])
 ax.set_xlabel('Date')
 ax.set_ylabel('Close Price')
 st.pyplot(fig)

 st.subheader("Trading Volume Over Time")
 fig, ax = plt.subplots()
 ax.plot(data.index, data['Volume'])
 ax.set_xlabel('Date')
 ax.set_ylabel('Volume')
 st.pyplot(fig)

 st.subheader("Correlation Heatmap")
 fig, ax = plt.subplots()
 sns.heatmap(data.corr(), annot=True, ax=ax)
 st.pyplot(fig)


# Prepare features and target for close price
X = data.index.map(pd.Timestamp.toordinal).values.reshape(-1, 1)
y_close_price = data['Close '].values

# Split data for close price
X_train_close, X_test_close, y_train_close, y_test_close = train_test_split(X, y_close_price, test_size=0.2, shuffle=False)

# Train model for close price
close_price_model = LinearRegression()
close_price_model.fit(X_train_close, y_train_close)

# Save close price model
joblib.dump(close_price_model, 'reliance_close_price_model.pkl')

# Prepare features and target for volume
y_volume = data['Volume'].values

# Split data for volume
X_train_volume, X_test_volume, y_train_volume, y_test_volume = train_test_split(X, y_volume, test_size=0.2, shuffle=False)

# Train model for volume
volume_model = LinearRegression()
volume_model.fit(X_train_volume, y_train_volume)

# Save volume model
joblib.dump(volume_model, 'reliance_volume_model.pkl')

try:
    close_price_model = joblib.load('reliance_close_price_model.pkl')
    volume_model = joblib.load('reliance_volume_model.pkl')
except FileNotFoundError as e:
    st.error(f"Model file not found: {e}")
    st.stop()

st.title("Reliance Industries Stock Data Prediction")

# Date input
prediction_date = st.date_input("Enter a date for prediction (2024-2029):", value=pd.to_datetime('2024-01-01'))

# Ensure date is within the specified range
if prediction_date < pd.to_datetime('2024-01-01') or prediction_date > pd.to_datetime('2029-12-31'):
    st.error("Please select a date between 2024 and 2029.")
else:
    prediction_date_ordinal = np.array([[prediction_date.toordinal()]])

    # Button to predict closing price
    if st.button('Predict Close Price'):
        predicted_close_price = close_price_model.predict(prediction_date_ordinal)
        st.subheader(f"Predicted Close Price for {prediction_date}:")
        st.write(f"{predicted_close_price[0]:.2f}")
        st.balloons() 
    # Button to predict volume
    if st.button('Predict Volume'):
        predicted_volume = volume_model.predict(prediction_date_ordinal)
        st.subheader(f"Predicted Volume for {prediction_date}:")
        st.write(f"{predicted_volume[0]:.2f}")
        st.balloons() 