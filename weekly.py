import joblib
import pandas as pd
import streamlit as st
import yfinance as yf

model = joblib.load("DayOfWeek_LinearRegression.pkl")
linear_model = joblib.load("Future_LinearRegression.pkl")
gradient_model = joblib.load("Future_GradientBoostingRegressor.pkl")
random_forest_model = joblib.load("Future_RandomForestRegressor.pkl")


def load_data():
    combined_df = pd.read_csv("combined_df.csv")

    # Convert 'Date' column to datetime
    combined_df["Date"] = pd.to_datetime(combined_df["Date"])

    # Add day of the week
    combined_df["DayOfWeek"] = combined_df["Date"].dt.dayofweek  # Monday=0, Sunday=6

    return combined_df


data = load_data()

st.title("Stock Analysis and Prediction")
st.sidebar.header("Options")

# Select the range of years using a slider
st.sidebar.subheader("Year Range for Prediction")
min_year = int(data["Year"].min())
max_year = int(data["Year"].max())

year_range = st.sidebar.slider(
    "Select Year Range", min_year, max_year, (min_year, max_year)
)
filtered_data = data[(data["Year"] >= year_range[0]) & (data["Year"] <= year_range[1])]


def fetch_stock_data(symbol, period="1y"):
    """
    Fetch historical stock data for the given symbol.
    """
    try:
        stock_data = yf.download(symbol, period=period)
        stock_data.reset_index(inplace=True)
        return stock_data
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        return None


def preprocess_data(df):
    """
    Add lagged features, moving averages, and other required columns for prediction.
    """
    # Use 'Close' instead of 'Adj Close'
    df["DailyReturn"] = df["Close"].pct_change()
    df["Lag_1"] = df["Close"].shift(1)
    df["Lag_2"] = df["Close"].shift(2)
    df["Lag_3"] = df["Close"].shift(3)
    df["MA_5"] = df["Close"].rolling(window=5).mean()
    df["MA_10"] = df["Close"].rolling(window=10).mean()
    df.dropna(inplace=True)
    return df


# User inputs the stock symbol
stock_symbol = st.sidebar.text_input("Enter Stock Symbol (e.g., AAPL, MSFT)", "AAPL")

# Prediction button
if st.sidebar.button("Predict Future Prices"):
    st.subheader(f"Prediction for {stock_symbol}")

    # Fetch stock data
    st.write(f"Fetching data for {stock_symbol}...")
    stock_data = fetch_stock_data(stock_symbol, period="1y")

    if stock_data is not None and not stock_data.empty:
        st.write(f"Data fetched for {stock_symbol}:")
        st.dataframe(stock_data.tail())

        # Preprocess the data
        stock_data = preprocess_data(stock_data)
        if stock_data.empty:
            st.error("Not enough data to preprocess and make predictions.")
        else:
            # Prepare the latest data for prediction
            recent_data = stock_data.iloc[-14:]  # Last 14 days for 2-week predictions
            input_features = recent_data[
                [
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
            ]

            # Make predictions using the trained models
            linear_predictions = linear_model.predict(input_features)
            gradient_predictions = gradient_model.predict(input_features)
            random_forest_predictions = random_forest_model.predict(input_features)

            # Display the results
            predictions_df = pd.DataFrame(
                {
                    "Date": pd.date_range(
                        start=recent_data["Date"].iloc[-1], periods=14, freq="D"
                    ),
                    "Linear_Prediction": linear_predictions,
                    "Gradient_Prediction": gradient_predictions,
                    "RandomForest_Prediction": random_forest_predictions,
                }
            )
            st.write("Predictions for the next 2 weeks:")
            st.dataframe(predictions_df)

            # Allow downloading the predictions
            st.download_button(
                label="Download Predictions",
                data=predictions_df.to_csv(index=False),
                file_name=f"{stock_symbol}_2_week_predictions.csv",
                mime="text/csv",
            )
    else:
        st.error(
            "Failed to fetch data or no data available for the given stock symbol."
        )


# Best day of the week analysis
if st.sidebar.checkbox("Best Day of the Week to Trade"):
    st.subheader("Best Day of the Week to Trade")
    day_of_week_avg = (
        filtered_data.groupby("DayOfWeek")["DailyReturn"].mean().reset_index()
    )
    day_of_week_avg["Day"] = day_of_week_avg["DayOfWeek"].map(
        {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday"}
    )
    best_day = day_of_week_avg.loc[day_of_week_avg["DailyReturn"].idxmax()]
    st.write(day_of_week_avg[["Day", "DailyReturn"]])
    st.write(
        f"**Best Day to Trade Stocks:** {best_day['Day']} with an average return of {best_day['DailyReturn']:.2f}%."
    )

# Yearly analysis
if st.sidebar.checkbox("Best Year for Stock Returns"):
    st.subheader("Best Year for Stock Returns")
    yearly_avg = filtered_data.groupby("Year")["DailyReturn"].mean().reset_index()
    best_year = yearly_avg.loc[yearly_avg["DailyReturn"].idxmax()]
    st.write(yearly_avg)
    st.write(
        f"**Best Year for Stock Returns:** {int(best_year['Year'])} with an average return of {best_year['DailyReturn']:.2f}%."
    )

# Stock ticker performance
if st.sidebar.checkbox("Best Stock Ticker Analysis"):
    st.subheader("Best Stock Ticker Analysis")
    year = st.selectbox(
        "Select Year",
        sorted(filtered_data["Year"].unique()),
        key="stock_ticker_select_year",
    )
    day = st.selectbox(
        "Select Day of the Week",
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
        key="stock_ticker_select_day",
    )

    day_to_num = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4}
    filtered_day_data = filtered_data[
        (filtered_data["Year"] == year)
        & (filtered_data["DayOfWeek"] == day_to_num[day])
    ]

    if not filtered_day_data.empty:
        best_stock = (
            filtered_day_data.groupby("Stock")["DailyReturn"].mean().reset_index()
        )
        best_stock = best_stock.loc[best_stock["DailyReturn"].idxmax()]
        st.write(
            f"**Best Stock of {year} on {day}:** {best_stock['Stock']} with an average return of {best_stock['DailyReturn']:.2f}%."
        )
    else:
        st.write("No data available for the selected year and day.")

# Interactive predictions
if st.sidebar.checkbox("Predict Returns for Day of the Week"):
    st.subheader("Predict Returns for Day of the Week")

    # User inputs for lagged features and moving averages
    lag_features = [st.number_input(f"Lag_{i}", value=0.0) for i in range(1, 6)]
    ma_5 = st.number_input("MA_5", value=0.0)
    ma_10 = st.number_input("MA_10", value=0.0)

    # User input for day of the week
    day = st.selectbox(
        "Select Day of the Week",
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
        key="predict_returns_select_day",
    )
    day_encoded = [
        (
            1
            if i == ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"].index(day)
            else 0
        )
        for i in range(5)
    ]

    # Create input feature dictionary
    input_features = {f"Lag_{i}": lag_features[i - 1] for i in range(1, 6)}
    input_features.update({"MA_5": ma_5, "MA_10": ma_10})
    input_features.update({f"DayOfWeek_{i}": day_encoded[i] for i in range(5)})

    # Convert to DataFrame
    input_df = pd.DataFrame([input_features])

    # Predict and display result
    if st.button("Predict"):
        predicted_return = linear_model.predict(input_df)[0]
        st.write(f"Predicted Daily Return for {day}: {predicted_return:.2f}%")
