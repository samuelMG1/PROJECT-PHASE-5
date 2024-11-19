from flask import Flask, render_template, request, jsonify
from markupsafe import Markup
import numpy as np
import pandas as pd
import requests
import config
import torch
import io
from utils.fertilizer import fertilizer_dic
from utils.disease import disease_dic
from torchvision import transforms
from PIL import Image
from utils.model import ResNet9
import pickle
import cv2
import sympy

# ==============================================================================================

# Loading crop recommendation model

crop_recommendation_model_path = 'models/RandomForest.pkl'
crop_recommendation_model = pickle.load(
    open(crop_recommendation_model_path, 'rb'))
    
# Custom functions for calculations

def weather_fetch(city_name):
    """
    Fetch and return the temperature and humidity of a city.
    :param city_name: Name of the city
    :return: (temperature, humidity) or None if the request fails
    """
    api_key = config.weather_api_key
    base_url = "https://api.openweathermap.org/data/2.5/weather"
    
    complete_url = f"{base_url}?q={city_name}&appid={api_key}"
    response = requests.get(complete_url)
    x = response.json()

    if x["cod"] != "404":
        y = x["main"]

        temperature = round((y["temp"] - 273.15), 2)
        humidity = y["humidity"]
        return temperature, humidity
    else:
        return None

    
# ===============================================================================================

# ------------------------------------ FLASK APP -------------------------------------------------

app = Flask(__name__)


@app.route('/')
def home():
    title = 'Wakulima FarmTech - Home'
    return render_template('index.html', title=title)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array([data])  # Assuming `data` contains an array-like input
    prediction = crop_recommendation_model.predict(features)
    return jsonify({'prediction': prediction.tolist()})

@app.route('/crop-recommend')
def crop_recommend():
    title = 'Wakulima FarmTech - Crop Recommendation'
    return render_template('crop.html', title=title)

@app.route('/fertilizer')
def fertilizer_recommendation():
    title = 'Wakulima FarmTech - Fertilizer Suggestion'
    return render_template('fertilizer.html', title=title)

# ===============================================================================================

# RENDER PREDICTION PAGES

@app.route('/crop-predict', methods=['POST'])
def crop_prediction():
    title = 'Wakulima FarmTech - Crop Recommendation'

    if request.method == 'POST':
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['pottasium'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])
        city = request.form.get("city")

        weather_data = weather_fetch(city)
        if weather_data is not None:
            temperature, humidity = weather_data
            data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            prediction = crop_recommendation_model.predict(data)
            final_prediction = prediction[0]
            return render_template('crop-result.html', prediction=final_prediction, title=title)
        else:
            return render_template('try_again.html', title=title)


@app.route('/fertilizer-predict', methods=['POST'])
def fert_recommend():
    title = 'Wakulima FarmTech - Fertilizer Suggestion'

    crop_name = str(request.form['cropname'])
    N = int(request.form['nitrogen'])
    P = int(request.form['phosphorous'])
    K = int(request.form['pottasium'])

    df = pd.read_csv(r'app\data\fertilizer.csv')


    nr = df[df['Crop'] == crop_name]['N'].iloc[0]
    pr = df[df['Crop'] == crop_name]['P'].iloc[0]
    kr = df[df['Crop'] == crop_name]['K'].iloc[0]

    n = nr - N
    p = pr - P
    k = kr - K
    temp = {abs(n): "N", abs(p): "P", abs(k): "K"}
    max_value = temp[max(temp.keys())]

    if max_value == "N":
        if n < 0:
            key = 'NHigh'
        else:
            key = "Nlow"
    elif max_value == "P":
        if p < 0:
            key = 'PHigh'
        else:
            key = "Plow"
    else:
        if k < 0:
            key = 'KHigh'
        else:
            key = "Klow"
            
    response = Markup(str(fertilizer_dic[key]))
    return render_template('fertilizer-result.html', recommendation=response, title=title)

@app.route('/disease-predict')
def disease_prediction():
    title = 'Wakulima FarmTech - Disease Prediction'
    return render_template('disease.html', title=title)

if __name__ == '__main__':
    app.run(debug=True)
