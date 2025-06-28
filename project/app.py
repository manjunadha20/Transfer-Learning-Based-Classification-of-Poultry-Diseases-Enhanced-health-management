import numpy as np
import os
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)
model = tf.keras.models.load_model("model.h5")

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def output():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    f = request.files['file']
    if f.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    upload_folder = "static/uploads"
    os.makedirs(upload_folder, exist_ok=True)
    img_path = os.path.join(upload_folder, f.filename)
    f.save(img_path)
    img = load_img(img_path, target_size=(224, 224))
    image_array = img_to_array(img)
    image_array = np.expand_dims(image_array, axis=0)
    preds = model.predict(image_array)
    pred = np.argmax(preds, axis=1)[0]
    confidence = float(np.max(preds))
    index = ['Coccidiosis', 'Healthy', 'New Castle Disease', 'Salmonella']
    prediction = index[int(pred)]
    return jsonify({'prediction': prediction, 'confidence': confidence})

if __name__ == '__main__':
    app.run(debug=True)
