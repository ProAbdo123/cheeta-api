from flask import Flask, request, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np

app = Flask(__name__)
model = tf.keras.models.load_model('model.h5')
CLASS_NAMES = ['Cheetah', 'Not Cheetah']
IMG_SIZE = (224, 224)

@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files['image']
    img = Image.open(image_file).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_index = int(np.argmax(prediction))
    predicted_label = CLASS_NAMES[predicted_index]
    confidence = float(prediction[0][predicted_index])

    return jsonify({
        "prediction": predicted_label,
        "confidence": round(confidence, 4)
    })
