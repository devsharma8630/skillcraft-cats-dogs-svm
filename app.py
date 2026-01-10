from flask import Flask, render_template, request
import cv2
import numpy as np
import joblib
from skimage.feature import hog

app = Flask(__name__)

model = joblib.load("svm_model.pkl")
scaler = joblib.load("scaler.pkl")
IMG_SIZE = 64

@app.route('/', methods=['GET','POST'])
def home():
    result = ""
    if request.method == 'POST':
        file = request.files['image']
        file.save("upload.jpg")

        img = cv2.imread("upload.jpg")
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        features = hog(img, pixels_per_cell=(8,8), cells_per_block=(2,2), block_norm='L2-Hys')
        features = scaler.transform([features])

        pred = model.predict(features)

        if pred[0] == 0:
            result = "🐱 CAT"
        else:
            result = "🐶 DOG"

    return render_template("index.html", result=result)

app.run(debug=True)
