from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import os
app = Flask(__name__, template_folder='my_templates')


# Load the trained model
model = tf.keras.models.load_model("poultry_disease_model.h5")


# Class labels (same order as model training)
class_labels = ['Coccidiosis', 'Healthy', 'Newcastle', 'Salmonella']

# Treatment advice
treatment_advice = {
    'Coccidiosis': "Treat with anticoccidial drugs (e.g., amprolium). Maintain dry litter and proper sanitation.",
    'Healthy': "No disease detected. Keep monitoring regularly.",
    'Newcastle': "Vaccinate healthy birds. Isolate infected birds. Disinfect the environment thoroughly.",
    'Salmonella': "Use antibiotics like enrofloxacin (under vet supervision). Ensure clean feed and water supply."
}

# Image settings
img_height, img_width = 224, 224

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return render_template("index.html", disease="❌ No file uploaded.", treatment="")

    file = request.files['file']
    if file.filename == '':
        return render_template("index.html", disease="❌ No file selected.", treatment="")

    try:
        image = Image.open(file).convert("RGB")
        image = image.resize((img_width, img_height))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        predictions = model.predict(image_array)
        predicted_class = class_labels[np.argmax(predictions)]
        confidence = round(100 * np.max(predictions), 2)
        treatment = treatment_advice.get(predicted_class, "No treatment advice available.")

        disease_result = f"{predicted_class} ({confidence}%)"

        return render_template("index.html", disease=disease_result, treatment=treatment)

    except Exception as e:
        return render_template("index.html", disease="⚠ Error processing image.", treatment=str(e))

if __name__ == "__main__":
    app.run(debug=True)