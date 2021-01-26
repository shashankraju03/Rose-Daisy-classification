from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import cv2
# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model


# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'model.h5'

# Load your trained model
model = load_model(MODEL_PATH)
        

CATEGORIES = ['daisy','rose']

def image(path):
    img = cv2.imread(path)
    new_arr = cv2.resize(img, (224, 224))
    new_arr = np.array(new_arr)
    new_arr = new_arr.reshape(-1, 224, 224, 3)
    return new_arr
def model_predict(img_path, model):
    preds = model.predict(image(img_path))
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)

        # Process your result for human
        result = CATEGORIES[preds.argmax()]              
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)

