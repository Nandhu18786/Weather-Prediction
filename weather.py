import requests
import pandas as pd
import numpy as np
from pymongo import MongoClient
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import streamlit as st
from dateutil import parser
from sklearn.impute import SimpleImputer

# Set up MongoDB connection
def connect_to_mongo():
    client = MongoClient('mongodb://localhost:27017/')
    db = client['weather_db']
    collection = db['weather_data']
    return collection

# Get the latest date for which data exists in MongoDB
def get_latest_date_for_city(collection, city_name):
    latest_record = collection.find({"city": city_name}).sort("date", -1).limit(1)
    latest_record = list(latest_record)
    if latest_record:
        date_value = latest_record[0]['date']
        return date_value.date() if isinstance(date_value, (datetime, pd.Timestamp)) else parser.isoparse(date_value).date()
    return None

# Get latitude and longitude for a city using OpenCage Geocoding API
def get_lat_long(city_name, api_key):
    url = "https://api.opencagedata.com/geocode/v1/json"
    params = {"q": city_name, "key": api_key, "limit": 1}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        if data['results']:
            return data['results'][0]['geometry']['lat'], data['results'][0]['geometry']['lng']
        st.error(f"No results found for city: {city_name}")
    else:
        st.error(f"Error fetching location data: {response.status_code} - {response.text}")
    return None, None

# Fetch and store weather data using Open-Meteo API
def fetch_weather_data(latitude, longitude, city_name):
    collection = connect_to_mongo()
    latest_date = get_latest_date_for_city(collection, city_name)
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=180)
    
    if latest_date:
        start_date = latest_date + timedelta(days=1)
        st.info(f"City '{city_name}' already exists in the database. Fetching data from {start_date} onwards.")
    else:
        st.info(f"City '{city_name}' not found in the database. Fetching data from the last 6 months.")
    
    url = "https://archive-api.open-meteo.com/v1/archive"

    current_date = start_date
    while current_date <= end_date:
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": current_date.strftime("%Y-%m-%d"),
            "end_date": current_date.strftime("%Y-%m-%d"),
            "hourly": "temperature_2m,weathercode"
        }
        
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            try:
                data = response.json()
                if 'hourly' in data and 'temperature_2m' in data['hourly']:
                    hourly_data = data['hourly']
                    temperatures = hourly_data['temperature_2m']
                    weather_codes = hourly_data['weathercode']
                    timestamps = hourly_data['time']

                    for i in range(len(temperatures)):
                        weather_record = {
                            "city": city_name,
                            "date": parser.isoparse(timestamps[i]),
                            "temperature": temperatures[i],
                            "weather_code": weather_codes[i]
                        }
                        collection.insert_one(weather_record)
                        #st.success(f"Weather data for {timestamps[i]} stored successfully!")
                else:
                    st.warning(f"No hourly data found for {current_date}. Response: {data}")

            except ValueError:
                st.error("Error: Response is not in JSON format.")
                return
        else:
            st.error(f"Error fetching data for {current_date}: {response.status_code} - {response.text}")
        
        current_date += timedelta(days=1)

    null_temperature_count = collection.count_documents({"city": city_name, "temperature": None})
    if null_temperature_count > 0:
        collection.delete_many({"city": city_name, "temperature": None})
        st.warning(f"Deleted {null_temperature_count} records with null temperature values.")
    else:
        st.info("No null temperature records found.")

    st.success("Weather data fetching completed.")

# Fetch historical weather data from MongoDB
def fetch_historical_data(collection, city_name):
    cursor = collection.find({"city": city_name})
    data = list(cursor)
    
    if not data:
        return pd.DataFrame()  # Return empty DataFrame if no data
    
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='date')
    
    return df

# Preprocess the data
def preprocess_data(df):
    df['hour'] = df['date'].dt.hour
    X = df[['temperature', 'hour', 'weather_code']]
    y = df['temperature'].shift(-1).dropna()
    X = X[:-1].dropna()
    y = y.dropna()

    if len(X) != len(y):
        X = X.head(len(y))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

# Train ensemble model
def train_ensemble_model(X_train, y_train):
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
    lr = LinearRegression()
    ensemble_model = VotingRegressor([('rf', rf), ('gb', gb), ('lr', lr)])
    ensemble_model.fit(X_train, y_train)
    
    return ensemble_model

# Predict temperature
def predict_future_temperature(ensemble_model, X_future, scaler):
    # Impute missing values in X_future if any
    imputer = SimpleImputer(strategy='mean')
    X_future_imputed = imputer.fit_transform(X_future)

    # Scale X_future
    X_future_scaled = scaler.transform(X_future_imputed)

    # Make predictions
    predictions = ensemble_model.predict(X_future_scaled)
    
    return predictions

# Fetch actual temperature from Open-Meteo API
def fetch_api_temperature(latitude, longitude, future_time):
    url = "https://ensemble-api.open-meteo.com/v1/ensemble"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "temperature_2m",
        "models": "icon_seamless",
        "start": int(future_time.timestamp()),  # start time in UNIX timestamp
        "end": int((future_time + pd.Timedelta(hours=1)).timestamp())  # end time in UNIX timestamp
    }
    response = requests.get(url, params=params)
    response_data = response.json()
    
    if 'hourly' in response_data and 'temperature_2m' in response_data['hourly']:
        api_temperature = response_data['hourly']['temperature_2m'][0]
        return api_temperature
    return None

import altair as alt
import pandas as pd
from datetime import datetime, timedelta
# Update the predict_temperature_intervals function to accept df as a parameter
def predict_temperature_intervals(ensemble_model, initial_temp, future_times, scaler, df):
    predictions = []
    
    # Create future data for each time interval and predict temperature
    for future_time in future_times:
        future_data = pd.DataFrame({
            'temperature': [initial_temp],  # Use the last known temperature as initial
            'hour': [future_time.hour],
            'weather_code': [df['weather_code'].iloc[-1]]  # Keep the weather code static
        })
        
        predicted_temp = predict_future_temperature(ensemble_model, future_data, scaler)[0]
        predictions.append((future_time, predicted_temp))
    
    return predictions
def classify_temperature(temp):
    if temp > 25:
        return "Clear / Sunny"
    elif 15 <= temp <= 25:
        return "Partly Cloudy"
    elif 10 <= temp < 15:
        return "Cloudy / Overcast"
    elif 5 <= temp < 10:
        return "Cool / Rainy"
    else:
        return "Cold / Snowy"
    
# Streamlit App
def main():
    st.title("Weather Data Fetcher and Temperature Prediction App")
    
    geocoding_api_key = "176ae02dc4ce4d61901a0d1e34a36a7a"  # Replace with your OpenCage API key
    latitude, longitude = None, None

    # Fetch weather data section
    city_name = st.text_input("Enter the city name:")
    
    if st.button("Fetch Weather Data"):
        if city_name:
            latitude, longitude = get_lat_long(city_name, geocoding_api_key)
            if latitude is not None and longitude is not None:
                fetch_weather_data(latitude, longitude, city_name)
            else:
                st.error("Failed to fetch coordinates for the city. Please check the city name.")
        else:
            st.error("Please enter a city name.")

    # Temperature prediction section with visualization
    st.write("Enter the city and time interval for temperature prediction.")
    
    city_name_temp = st.text_input("City Name for Temperature Prediction", "")
    time_unit = st.radio("Select Time Unit", ("Hours", "Minutes"))
    future_time_interval = st.number_input(f"Select Future {time_unit} Interval", min_value=1, max_value=23 if time_unit == "Hours" else 59)

    if st.button("Predict Temperature with Visualization"):
        if city_name_temp:
            # Fetch historical weather data from MongoDB
            collection = connect_to_mongo()
            df = fetch_historical_data(collection, city_name_temp)

            if not df.empty:
                # Check if latitude and longitude were assigned
                if latitude is None or longitude is None:
                    latitude, longitude = get_lat_long(city_name_temp, geocoding_api_key)
                    if latitude is None or longitude is None:
                        st.error("Could not fetch coordinates for the city.")
                        return
                
                # Preprocess and train the model
                X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
                ensemble_model = train_ensemble_model(X_train, y_train)
                
                # Set up times for future predictions
                intervals = 5
                future_times = [
                    datetime.now() + timedelta(
                        hours=(future_time_interval * i) if time_unit == "Hours" else 0,
                        minutes=(future_time_interval * i) if time_unit == "Minutes" else 0
                    ) for i in range(1, intervals + 1)
                ]
                
                # After generating predictions, classify them
                predictions = predict_temperature_intervals(
                    ensemble_model,
                    initial_temp=df['temperature'].iloc[-1],
                    future_times=future_times,
                    scaler=scaler,
                    df=df  # Pass df as an argument
                )

                # Create DataFrame for visualization
                prediction_df = pd.DataFrame(predictions, columns=['Time', 'Predicted Temperature'])
                prediction_df['Time'] = pd.to_datetime(prediction_df['Time'])

                #Classify the predicted temperatures and add to DataFrame
                prediction_df['Classification'] = prediction_df['Predicted Temperature'].apply(classify_temperature)

                # Display prediction values in a chart
                st.write(f"Predicted Temperatures for the next {intervals} intervals of {future_time_interval} {time_unit.lower()}s:")

                # Plot using Altair with text annotation for temperature values
                line_chart = alt.Chart(prediction_df).mark_line(color='blue').encode(
                x=alt.X('Time:T', title='Future Time'),
                y=alt.Y('Predicted Temperature:Q', title='Temperature (°C)'),
                tooltip=['Time:T', 'Predicted Temperature:Q', 'Classification:N']
                ).properties(
                    title="Temperature Predictions Over Time"
                )

                # Add text labels to show each temperature value on the line
                text_labels = alt.Chart(prediction_df).mark_text(
                align='left',
                dx=5,  # Offset for text labels
                dy=-10,  # Offset vertically above points
                fill='white'  # Set the text color to white
                ).encode(
                    x='Time:T',
                    y='Predicted Temperature:Q',
                    text=alt.Text('Predicted Temperature:Q', format='.1f')  # Format to 1 decimal place
                )

                # Combine the line chart and text labels
                st.altair_chart(line_chart + text_labels, use_container_width=True)

                # Summary of classification
                st.markdown("### Summary of Classification")
                classification_summary = """
                    - *Above 25°C*: Clear / Sunny
                    - *15°C to 25°C*: Partly Cloudy
                    - *10°C to 15°C*: Cloudy / Overcast
                    - *5°C to 10°C*: Cool / Rainy
                    - *Below 5°C*: Cold / Snowy
                """
                st.markdown(classification_summary)

                # Display the classification results for the intervals
                st.markdown("### Predictions with Classifications")
                for index, row in prediction_df.iterrows():
                    st.write(f"*Time:* {row['Time']}, *Predicted Temperature:* {row['Predicted Temperature']}°C, *Classification:* {row['Classification']}")


            else:
                st.error(f"No historical weather data found for {city_name_temp}. Please fetch weather data first.")
        else:
            st.error("Please enter a city name for temperature prediction.")

if __name__ == "__main__":
    main()
