import os

import joblib
import pandas as pd
import streamlit as st

# Load the trained model
model = joblib.load("DayOfWeek_LinearRegression.pkl")


@st.cache_resource
def load_data():
    import kagglehub

    dataset_path = kagglehub.dataset_download(
        "borismarjanovic/price-volume-data-for-all-us-stocks-etfs",
    )

    print("Path to dataset files:", dataset_path)

    dataset_path += "/Stocks"

    # List to store individual DataFrames
    dataframes = []

    # Iterate through all files in the directory
    for file_name in os.listdir(dataset_path):
        # Only process .txt files
        if file_name.endswith(".txt"):
            file_path = os.path.join(dataset_path, file_name)

            if len(open(file_path).read()) == 0:
                continue

            # Read the file into a DataFrame, assuming the first line is the header
            df = pd.read_csv(file_path)

            # Add a 'Stock' column to store the stock symbol (derived from file name)
            df["Stock"] = file_name.split(".")[0]

            # Drop the 'OpenInt' column since it's not necessary for the analysis
            df = df.drop(columns=["OpenInt"])

            # Append to the list of DataFrames
            dataframes.append(df)

    # Combine all DataFrames into one
    combined_df = pd.concat(dataframes, ignore_index=True)

    # Display a summary of the combined DataFrame
    print(f"Combined DataFrame shape: {combined_df.shape}")
    print(combined_df.head())

    # Check for missing values
    print(combined_df.isnull().sum())

    # Check for duplicates
    duplicates = combined_df.duplicated().sum()
    print(f"Number of duplicate rows: {duplicates}")

    # If duplicates exist, drop them
    combined_df = combined_df.drop_duplicates()

    # Convert 'Date' column to datetime
    combined_df["Date"] = pd.to_datetime(combined_df["Date"])

    print(combined_df.dtypes)

    # Add derived features
    combined_df["Year"] = combined_df["Date"].dt.year
    combined_df["Month"] = combined_df["Date"].dt.month
    combined_df["DayOfWeek"] = combined_df["Date"].dt.day_name()
    combined_df["DailyReturn"] = (
        combined_df.groupby("Stock")["Close"].pct_change() * 100
    )

    election_years = [
        1962,
        1964,
        1966,
        1968,
        1970,
        1972,
        1974,
        1976,
        1978,
        1980,
        1982,
        1984,
        1986,
        1988,
        1990,
        1992,
        1994,
        1996,
        1998,
        2000,
        2002,
        2004,
        2006,
        2008,
        2010,
        2012,
        2014,
        2016,
        2018,
        2020,
        2022,
        2024,
    ]

    combined_df["IsElectionYear"] = combined_df["Year"].apply(
        lambda x: 1 if x in election_years else 0
    )

    # Add day of the week
    combined_df["DayOfWeek"] = combined_df["Date"].dt.dayofweek  # Monday=0, Sunday=6

    return combined_df


data = load_data()

# App layout
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
