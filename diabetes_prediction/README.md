## End to End Machine Learning Project
# Diabetes Prediction Web Application

This project is a web application that predicts the likelihood of diabetes based on user-provided health data. It uses a machine learning model trained on health-related features to provide a probability of diabetes. The app is built with Flask and a trained model is deployed for predictions.

## Project Overview

The diabetes prediction model uses a dataset of health parameters to predict diabetes likelihood. The model considers factors such as age, BMI, blood glucose level, and other health indicators. Users can enter their information via a web form, and the application provides a prediction along with a probability score.

## Features

- **User-friendly Interface**: Collects input data from the user in a structured form.
- **Machine Learning Model**: Uses a pre-trained model to predict the likelihood of diabetes.
- **Real-time Predictions**: Returns a probability score of diabetes based on input data.

## Technologies Used

- **Backend**: Python, Flask
- **Frontend**: HTML, CSS (TailwindCSS)
- **Machine Learning**: Scikit-Learn, Pandas
- **Deployment**: Joblib for model serialization

## Getting Started

To get a local copy up and running, follow these steps:

### Prerequisites

- Python 3.8 or above
- Flask
- Scikit-Learn
- Pandas
- Joblib

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/biniyam93/diabetes_prediction.git
    cd diabetes_prediction
    ```

2. Create a virtual environment:
    ```bash
    python3 -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

4. Place the pre-trained model and preprocessing pipeline in the `artifacts` folder:
    - `model.pkl`: The trained machine learning model.
    - `preprocessor.pkl`: The preprocessing pipeline used on the input data.

### Running the Application

1. Start the Flask application:
    ```bash
    python app.py
    ```

2. Open your browser and navigate to `http://127.0.0.1:5000` to access the app.

## Project Structure


## Usage

1. Open the application in your browser.
2. Enter the required health parameters into the form, including:
   - Gender
   - Age
   - Hypertension
   - Heart Disease
   - Smoking History
   - BMI
   - HbA1c Level
   - Blood Glucose Level
3. Click "Predict" to get the diabetes prediction and probability score.

## Example Prediction

Sample input data and expected prediction output:
- **Input**: `{'gender': 'Male', 'age': 45, 'hypertension': 1, 'heart_disease': 0, 'smoking_history': 'Former', 'bmi': 30.5, 'HbA1c_level': 6.8, 'blood_glucose_level': 140}`
- **Output**: `Prediction: Positive | Diabetes Probability: 0.78`

## Model and Data

The model was trained on a diabetes dataset using various health parameters to classify the likelihood of diabetes. The data was preprocessed and balanced to improve model accuracy, and hyperparameter tuning was performed for optimal performance.

## Acknowledgements

- Scikit-Learn for providing machine learning tools.

## Contact

Project Link: [GitHub Repository](https://github.com/biniyam93/diabetes_prediction)
