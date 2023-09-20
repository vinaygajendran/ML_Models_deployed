import numpy as np
import pandas as pd
from flask import Flask, request, render_template, jsonify
import pickle

app = Flask(__name__)

# Load the saved Random Forest model
loaded_model = pickle.load(open('concrete_strength', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the HTML form
        material_quantity = float(request.form['material_quantity'])
        additive_catalyst = float(request.form['additive_catalyst'])
        plasticizer = float(request.form['plasticizer'])
        formulation_duration = float(request.form['formulation_duration'])

        # Create an input array for prediction
        input_data = np.array([[material_quantity, additive_catalyst, plasticizer, formulation_duration]])

        # Make predictions using the loaded model
        prediction = loaded_model.predict(input_data)

        # Pass the prediction result to the result.html template
        return render_template('result.html', prediction=prediction[0])
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)
