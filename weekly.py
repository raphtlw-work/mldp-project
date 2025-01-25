import joblib
import pandas as pd
import streamlit as st

# Load the trained model
model = joblib.load("DayOfWeek_LinearRegression.pkl")


def load_data():
    combined_df = pd.read_csv("combined_df.csv")

    # Convert 'Date' column to datetime
    combined_df["Date"] = pd.to_datetime(combined_df["Date"])

    # Add day of the week
    combined_df["DayOfWeek"] = combined_df["Date"].dt.dayofweek  # Monday=0, Sunday=6

    return combined_df


data = load_data()

st.title("Stock Analysis: Best Days and Years to Trade")
st.sidebar.header("Analysis Options")

# User selection for day-of-week analysis
if st.sidebar.checkbox("Best Day of the Week to Trade"):
    st.subheader("Best Day of the Week to Trade")
    day_of_week_avg = data.groupby("DayOfWeek")["DailyReturn"].mean().reset_index()
    day_of_week_avg["Day"] = day_of_week_avg["DayOfWeek"].map(
        {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday"}
    )
    best_day = day_of_week_avg.loc[day_of_week_avg["DailyReturn"].idxmax()]
    st.write(day_of_week_avg[["Day", "DailyReturn"]])
    st.write(
        f"**Best Day to Trade Stocks:** {best_day['Day']} with an average return of {best_day['DailyReturn']:.2f}%."
    )

# User selection for yearly analysis
if st.sidebar.checkbox("Best Year for Stock Returns"):
    st.subheader("Best Year for Stock Returns")
    yearly_avg = data.groupby("Year")["DailyReturn"].mean().reset_index()
    best_year = yearly_avg.loc[yearly_avg["DailyReturn"].idxmax()]
    st.write(yearly_avg)
    st.write(
        f"**Best Year for Stock Returns:** {int(best_year['Year'])} with an average return of {best_year['DailyReturn']:.2f}%."
    )

# User selection for ticker performance
if st.sidebar.checkbox("Best Stock Ticker Analysis"):
    st.subheader("Best Stock Ticker Analysis")
    year = st.selectbox(
        "Select Year", sorted(data["Year"].unique()), key="best_stock_year"
    )
    day = st.selectbox(
        "Select Day of the Week",
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
        key="best_stock_day",
    )

    day_to_num = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4}
    filtered_data = data[
        (data["Year"] == year) & (data["DayOfWeek"] == day_to_num[day])
    ]

    if not filtered_data.empty:
        best_stock = filtered_data.groupby("Stock")["DailyReturn"].mean().reset_index()
        best_stock = best_stock.loc[best_stock["DailyReturn"].idxmax()]
        st.write(
            f"**Best Stock of {year} on {day}:** {best_stock['Stock']} with an average return of {best_stock['DailyReturn']:.2f}%."
        )
    else:
        st.write("No data available for the selected year and day.")

# User selection for interactive predictions
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
        key="predict_day",
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
        predicted_return = model.predict(input_df)[0]
        st.write(f"Predicted Daily Return for {day}: {predicted_return:.2f}%")
