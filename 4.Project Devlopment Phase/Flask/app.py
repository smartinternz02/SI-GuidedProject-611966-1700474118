from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd

model = pickle.load(open(r'C:\Users\kaler\Desktop\Main folder\temperature.pkl', 'rb'))
app = Flask(__name__)

# Define the feature names used in your model
names = ['CO2_room', 'Relative_humidity_room', 'Lighting_room', 'Meteo_Rain', 'Meteo_Wind',
         'Meteo_Sun_light_in_west_facade', 'Outdoor_relative_humidity_Sensor']

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict")
def predict():
    return render_template("predict.html")

@app.route('/output', methods=['POST', 'GET'])
def output():
    try:
        input_feature = [float(request.form.get(name, 0.0)) for name in names]
        data = pd.DataFrame([input_feature], columns=names)
        prediction = model.predict(data)
        indoor_temperature_prediction = prediction[0]  # Extract the predicted indoor temperature
        return render_template('predict.html', indoor_temperature_prediction=indoor_temperature_prediction)

    except Exception as e:
        print("Error:", e)
        return f"An error occurred: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
