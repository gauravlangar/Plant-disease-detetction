# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 12:36:26 2020

@author: souren
"""

from flask import Flask, render_template, request
import numpy as np
import pickle
import cv2
from keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import os
from werkzeug.utils import secure_filename
UPLOAD_FOLDER = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/motivation",methods=['GET'])
def motivation():
    return render_template("motivation.html")
@app.route("/algorithm",methods=['GET'])
def algorithm():
    return render_template("algorithm.html")
@app.route("/ourcontribution",methods=['GET'])
def ourcontribution():
     return render_template("contribution.html")
@app.route("/home",methods=['GET'])
def home():
    global model
    model = load_model("tomato_leaves.hdf5")
    return render_template('using.html')


@app.route('/uploadajax', methods = ['POST','GET'])


def upldfile():
    target = os.path.join(UPLOAD_FOLDER, 'static/')
    upload = os.path.join(target, 'uploads/')
    if not os.path.isdir(target):
        os.mkdir(target)
        os.mkdir(upload)
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        file.save(os.path.join(upload, filename))
        image = os.path.join(upload,filename)
        status = load_classification_model(image)
        return status
    else:
        return "Fail"

def convert_image_to_array(image_dir):
    default_image_size = tuple((256, 256))
    try:
        image = cv2.imread(image_dir)
        if image is not None :
            image = cv2.resize(image, default_image_size)
            return img_to_array(image)
        else :
            return np.array([])
    except Exception as e:
        print(f"Error : {e}")
        return None

def load_classification_model(image):
    image_array = convert_image_to_array(image)
    np_image_array = np.array(image_array, dtype=np.float16) / 225.0
    np_image_array = np_image_array.reshape((-1, 256, 256,3))
    pkl_file = open('label_transform.pkl', 'rb')
    image_labels = pickle.load(pkl_file)
    pkl_file.close()

    disease_name = image_labels.inverse_transform(model.predict(np_image_array))[0]
    #disease_name = disease_name[9:]
    return disease_name[9:]

if __name__ == "__main__":
    app.run()
