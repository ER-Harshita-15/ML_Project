from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle  # For loading .pkl files

application = Flask(__name__)
app = application

# Load the model and preprocessor from .pkl files
with open('artifacts/model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('artifacts/preprocessor.pkl', 'rb') as scaler_file:
    preprocessor = pickle.load(scaler_file)

## Route For Home Page
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        # Render the prediction form
        return render_template('home.html')
    
    else:
        # Collect data from form
        data = {
            'gender': request.form['gender'],
            'race_ethnicity': request.form['race_ethnicity'],
            'parental_level_of_education': request.form['parental_level_of_education'],
            'lunch': request.form['lunch'],
            'test_preparation_course': request.form['test_preparation_course'],
            'reading_score': float(request.form['reading_score']),
            'writing_score': float(request.form['writing_score'])
        }

        df = pd.DataFrame([data])  # wrap into a DataFrame

        # Apply the saved preprocessing pipeline
        transformed_input = preprocessor.transform(df)

        # Predict
        results = model.predict(transformed_input)
        print (results)

        return render_template('home.html', results=results[0])
if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)