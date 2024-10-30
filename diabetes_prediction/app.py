from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model and preprocessing pipeline
model = joblib.load('artifacts/model.pkl')
preprocessing = joblib.load('artifacts/preprocessor.pkl')

# Mapping for manual encoding
gender_mapping = {'Male': 0, 'Female': 1}  # Adjust based on your dataset
smoking_history_mapping = {'Never': 0, 'Former': 1, 'Current': 2}  # Example, adjust to actual categories

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract data from form
        data = {
            'gender': request.form['gender'],
            'age': float(request.form['age']),
            'hypertension': float(request.form['hypertension']),
            'heart_disease': float(request.form['heart_disease']),
            'smoking_history': request.form['smoking_history'],
            'bmi': float(request.form['bmi']),
            'HbA1c_level': float(request.form['HbA1c_level']),
            'blood_glucose_level': float(request.form['blood_glucose_level'])
        }

        # Debugging: Print form data to check values
        print("Form Data Received:", data)

        # Apply manual encoding for categorical features
        gender_mapping = {'Male': 0, 'Female': 1}  # Adjust based on your dataset
        smoking_history_mapping = {'Never': 0, 'Former': 1, 'Current': 2}  # Adjust as needed

        data['gender'] = gender_mapping.get(data['gender'], -1)
        data['smoking_history'] = smoking_history_mapping.get(data['smoking_history'], -1)

        # Debugging: Check encoded values
        print("Encoded Gender:", data['gender'])
        print("Encoded Smoking History:", data['smoking_history'])

        # Ensure valid encoding
        if -1 in (data['gender'], data['smoking_history']):
            missing_fields = []
            if data['gender'] == -1:
                missing_fields.append('gender')
            if data['smoking_history'] == -1:
                missing_fields.append('smoking_history')
            return jsonify({'error': f'Invalid value for fields: {", ".join(missing_fields)}'}), 400

        # Create DataFrame
        df = pd.DataFrame([data])

        # Apply preprocessing
        X_transformed = preprocessing.transform(df)

        # Predict using the model
        prediction = model.predict(X_transformed)
        prediction_proba = model.predict_proba(X_transformed)[:, 1]

        # Prepare the result
        result = {
            'prediction': 'Positive' if prediction[0] == 1 else 'Negative',
            'diabetes_probability': round(float(prediction_proba[0]), 4)
        }

        return jsonify(result)

    except ValueError as e:
        return jsonify({'error': f'Invalid input type: {e}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
