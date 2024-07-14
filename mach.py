#!/usr/bin/env python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def load_and_preprocess_data(filepath):
    # Load the data
    data = pd.read_csv(filepath)
    
    # Convert necessary columns to numeric
    data["date"] = pd.to_numeric(data["date"], errors='coerce')
    data["player1_elo"] = pd.to_numeric(data["player1_elo"], errors='coerce')
    data["player2_elo"] = pd.to_numeric(data["player2_elo"], errors='coerce')
    data = data.dropna()

    # Create additional features
    data["player1_color"] = 'white'
    data["player2_color"] = 'black'

    data["player1_color_code"] = data["player1_color"].astype("category").cat.codes
    data["player2_color_code"] = data["player2_color"].astype("category").cat.codes
    data["player2_code"] = data["player2_name"].astype("category").cat.codes
    data["event_type_code"] = data["event_type"].astype("category").cat.codes
    data["event_format_code"] = data["event_format"].astype("category").cat.codes
    data["target"] = (data["player1_points"] == 2).astype(int)
    
    # Define predictors and target
    predictors = ["player1_elo", "player2_elo", "event_type_code", "event_format_code", "date"]
    target = "target"

    # Split the data into train and test sets
    X = data[predictors]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    
    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

def create_model(input_shape):
    # Define the model
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(input_shape,)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

def main():
    # Load and preprocess the data
    filepath = "/mnt/data/matches.csv"
    X_train, X_test, y_train, y_test = load_and_preprocess_data(filepath)
    
    # Create the model
    model = create_model(X_train.shape[1])
    
    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

    # Make predictions
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    
    # Evaluate the model
    precision = precision_score(y_test, y_pred)
    print(f"Precision: {precision}")

if __name__ == "__main__":
    main()