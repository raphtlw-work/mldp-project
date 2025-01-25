import joblib
import pandas as pd
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


# Load the combined dataset
@st.cache
def load_data():
    return pd.read_csv("combined_stock_data.csv")


# Load the model dynamically based on user input
@st.cache(allow_output_mutation=True)
def load_model(model_type):
    # Feature engineering (can be done outside if already processed)
    combined_df = load_data()

    # Convert 'Date' to datetime format and feature engineering
    combined_df["Date"] = pd.to_datetime(combined_df["Date"])
    combined_df["Year"] = combined_df["Date"].dt.year
    combined_df["Month"] = combined_df["Date"].dt.month
    combined_df["DayOfWeek"] = combined_df["Date"].dt.day_name()
    combined_df["DailyReturn"] = (
        combined_df.groupby("Stock")["Close"].pct_change() * 100
    )

    # Add lag features
    for lag in range(1, 6):
        combined_df[f"Lag_{lag}"] = combined_df.groupby("Stock")["Close"].shift(lag)

    # Add moving averages
    combined_df["MA_5"] = (
        combined_df.groupby("Stock")["Close"]
        .rolling(window=5)
        .mean()
        .reset_index(0, drop=True)
    )
    combined_df["MA_10"] = (
        combined_df.groupby("Stock")["Close"]
        .rolling(window=10)
        .mean()
        .reset_index(0, drop=True)
    )

    # Drop missing values after lagging
    combined_df = combined_df.dropna()

    # Select features and target
    features = [
        "Open",
        "High",
        "Low",
        "Volume",
        "Lag_1",
        "Lag_2",
        "Lag_3",
        "MA_5",
        "MA_10",
    ]
    target = "DailyReturn"

    X = combined_df[features]
    y = combined_df[target]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train and evaluate the model based on the selected model_type
    if model_type == "Linear Regression":
        model = LinearRegression()
    elif model_type == "Random Forest Regressor":
        model = RandomForestRegressor(random_state=42, n_estimators=100, n_jobs=-1)
    elif model_type == "Gradient Boosting Regressor":
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)

    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)

    # Save the model to a file (optional)
    joblib.dump(model, f'{model_type.replace(" ", "")}.pkl')

    return model, mae


# Streamlit App
st.title("Stock Market Analysis & Prediction")

# Sidebar Parameters
st.sidebar.header("Interactive Parameters")

# Model Selection
model_type = st.sidebar.selectbox(
    "Choose the model to train and predict:",
    ["Linear Regression", "Random Forest Regressor", "Gradient Boosting Regressor"],
)

# Load and train the selected model
st.sidebar.write(f"Training {model_type} model...")
model, mae = load_model(model_type)

# Display model performance
st.write(f"Mean Absolute Error (MAE) for {model_type}: {mae:.2f}")

# Display interactive prediction form
st.header("Stock Price Prediction")
open_price = st.number_input("Open Price", min_value=0.0, value=100.0)
high_price = st.number_input("High Price", min_value=0.0, value=105.0)
low_price = st.number_input("Low Price", min_value=0.0, value=95.0)
volume = st.number_input("Volume", min_value=0, value=100000)
lag_1 = st.number_input("Lag 1 (Previous Day Close Price)", min_value=0.0, value=100.0)
lag_2 = st.number_input("Lag 2 (2 Days Ago Close Price)", min_value=0.0, value=100.0)
lag_3 = st.number_input("Lag 3 (3 Days Ago Close Price)", min_value=0.0, value=100.0)
ma_5 = st.number_input("5-Day Moving Average", min_value=0.0, value=100.0)
ma_10 = st.number_input("10-Day Moving Average", min_value=0.0, value=100.0)

# Form for prediction
input_data = [
    [open_price, high_price, low_price, volume, lag_1, lag_2, lag_3, ma_5, ma_10]
]

# Button for prediction
if st.button("Predict Future Return"):
    prediction = model.predict(input_data)
    st.write(f"Predicted Daily Return: {prediction[0]:.2f}%")
