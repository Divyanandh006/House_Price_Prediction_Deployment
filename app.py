from flask import Flask, request, jsonify
import pandas as pd
import numpy as np # Import numpy for boolean operations if needed
import joblib
# Initialize the Flask application
app = Flask(__name__)

# Load the trained model from the file
loaded_rf_regressor = joblib.load('random_forest_regressor_model.joblib')

print("Random Forest Regressor model loaded successfully.")
# Define the expected feature columns and their data types from X_train
# This helps in creating a consistent input DataFrame
feature_columns = X_train.columns.tolist()
feature_dtypes = X_train.dtypes.to_dict()

@app.route('/predict', methods=['POST'])
def predict_house_price():
    data = request.get_json()
    if data is None:
        return jsonify({'error': 'No JSON data provided'}), 400

    # Initialize a dictionary to hold processed data for a single row
    processed_input = {}

    # Initialize all feature columns with default values and correct dtypes
    for col in feature_columns:
        if col in feature_dtypes:
            if pd.api.types.is_bool_dtype(feature_dtypes[col]):
                processed_input[col] = False
            elif pd.api.types.is_numeric_dtype(feature_dtypes[col]):
                processed_input[col] = 0 # Default for numerical
            else:
                processed_input[col] = None # Or appropriate default for other types

    # Process incoming data
    for key, value in data.items():
        if key in ['Id', 'Area', 'Bedrooms', 'Bathrooms', 'Floors', 'YearBuilt']:
            # Numerical features: direct assignment
            if key in processed_input:
                processed_input[key] = value
        elif key == 'Location':
            # One-hot encode 'Location'
            encoded_col = f'Location_{value}'
            if encoded_col in processed_input:
                processed_input[encoded_col] = True
        elif key == 'Condition':
            # One-hot encode 'Condition'
            encoded_col = f'Condition_{value}'
            if encoded_col in processed_input:
                processed_input[encoded_col] = True
        elif key == 'Garage':
            # One-hot encode 'Garage'
            # Assuming 'Garage' in input is a boolean (True/False) or 'Yes'/'No'
            if isinstance(value, bool):
                processed_input['Garage_Yes'] = value
            elif isinstance(value, str) and value.lower() == 'yes':
                processed_input['Garage_Yes'] = True
            else:
                processed_input['Garage_Yes'] = False

    # Create a DataFrame from the processed input, ensuring correct dtypes
    input_df = pd.DataFrame([processed_input]).astype(feature_dtypes)

    # Ensure the columns are in the exact order as X_train
    input_df = input_df[feature_columns]

    print(f"Processed input DataFrame:\n{input_df}")

    # Make prediction
    prediction = loaded_rf_regressor.predict(input_df)
    return jsonify({'predicted_price': prediction[0]})
if __name__ == "__main__":
    app.run(debug=True) 
