from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load the pre-trained model using pickle
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template("index.html")
    else:
        try:
            # Get input values from the form
            SepalLengthCm = float(request.form['SepalLengthCm'])
            SepalWidthCm = float(request.form['SepalWidthCm'])
            PetalLengthCm = float(request.form['PetalLengthCm'])
            PetalWidthCm = float(request.form['PetalWidthCm'])

            # Ensure that the input values are numerical
            if any(np.isnan([SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm])):
                raise ValueError("Invalid input. Please enter numerical values for all features.")

            # Make prediction using the pre-trained model
            input_data = np.array([[SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm]])
            prediction = int(model.predict(input_data)[0])  # Convert to integer

            # Map numerical prediction to corresponding species
            species_mapping = {0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'}
            
            # Handle cases where the prediction value is not in species_mapping
            predicted_species = species_mapping.get(prediction, 'Unknown Species')

            return render_template("index.html", prediction=f"Species: {predicted_species}")
        except ValueError as e:
            return render_template("index.html", error=str(e))

if __name__ == '__main__':
    app.run(debug=True)