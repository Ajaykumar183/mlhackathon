import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# Fetching the latest COVID-19 data
url = "https://disease.sh/v3/covid-19/countries/usa"
r = requests.get(url)
data = r.json()

# Extract relevant fields
covid_data = {
    "cases": data["cases"],
    "todayCases": data["todayCases"],
    "deaths": data["deaths"],
    "todayDeaths": data["todayDeaths"],
    "recovered": data["recovered"],
    "active": data["active"],
    "critical": data["critical"],
    "casesPerMillion": data["casesPerOneMillion"],
    "deathsPerMillion": data["deathsPerOneMillion"],
}

# Convert to Pandas DataFrame
df = pd.DataFrame([covid_data])
print(df)

# Plotting COVID-19 data for USA
labels = ["Total Cases", "Active Cases", "Recovered", "Deaths"]
values = [data["cases"], data["active"], data["recovered"], data["deaths"]]

plt.figure(figsize=(8,5))
plt.bar(labels, values, color=['blue', 'orange', 'green', 'red'])
plt.xlabel("Category")
plt.ylabel("Count")
plt.title("COVID-19 Data for USA")
plt.show()

# Generating random historical data (you can replace this with actual data)
np.random.seed(42)
historical_cases = np.random.randint(30000, 70000, size=30)  # Last 30 days cases
historical_deaths = np.random.randint(500, 2000, size=30)

df_historical = pd.DataFrame({"cases": historical_cases, "deaths": historical_deaths})
df_historical["day"] = range(1, 31)

print(df_historical.head())

# Split the data into features (X) and target (y)
X = df_historical[["day"]]
y = df_historical["cases"]

# Scale the features since SVR performs better with scaled data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize and train the Support Vector Regression model
svr_model = SVR(kernel="rbf")
svr_model.fit(X_train, y_train)

# Predict next day's cases
next_day = np.array([[31]])
next_day_scaled = scaler.transform(next_day)
predicted_cases = svr_model.predict(next_day_scaled)
print(f"Predicted cases for Day 31: {int(predicted_cases[0])}")

# Streamlit app for user input and prediction
st.title("COVID-19 Cases Prediction in USA")
st.write("Predicting COVID-19 cases for the next day based on historical data.")

# User Input
day_input = st.number_input("Enter day number (e.g., 31 for prediction)", min_value=1, max_value=100)

if st.button("Predict"):
    # Scale the input day
    day_input_scaled = scaler.transform([[day_input]])
    prediction = svr_model.predict(day_input_scaled)
    st.write(f"Predicted cases for day {day_input}: {int(prediction[0])}")
