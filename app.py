from flask import Flask, render_template, request, Markup
import numpy as np
import pandas as pd
import pickle
import sklearn

# importing model
model = pickle.load(open('model.pkl', 'rb'))
sc = pickle.load(open(''.pkl, 'rb'))
ms =pickle.load(open('model.ms', 'rb'))

# creating a flask app
app = Flask(__name__)

@app.route('/')
