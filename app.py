from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask_cors import CORS
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Load the trained model
MODEL_PATH = './models/models.h5'
if os.path.exists(MODEL_PATH):
    try:
        model = load_model(MODEL_PATH)
    except Exception as e:
        print("Error loading model:", e)
        model = None
else:
    print("Model not found at path:", MODEL_PATH)
    model = None

# Define a function to preprocess the uploaded image
def preprocess_image(img_path):
    try:
        img = image.load_img(img_path, target_size=(150, 150))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array, None
    except Exception as e:
        return None, str(e)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file.content_type not in ['image/jpeg', 'image/png']:
        return jsonify({'error': 'Only JPEG and PNG images are allowed'}), 400
    
    img_path = 'uploads/' + file.filename
    file.save(img_path)
    
    # Preprocess the uploaded image
    processed_img, error = preprocess_image(img_path)
    if error:
        os.remove(img_path)  # Delete the uploaded image if preprocessing fails
        return jsonify({'error': error}), 500
    
    # Make prediction using the model
    try:
        prediction = model.predict(processed_img)
        predicted_class = np.argmax(prediction)
        class_names = {0: 'Diseased Plant', 1: 'Healthy Plant'}
        predicted_class_name = class_names.get(predicted_class, 'Unknown')
        is_diseased = predicted_class == 0
        is_diseased_str = 'Yes' if is_diseased else 'No'
        os.remove(img_path)  # Delete the uploaded image after prediction
        return jsonify({'predicted_class': predicted_class_name, 'is_diseased': is_diseased_str}), 200
    except Exception as e:
        os.remove(img_path)  # Delete the uploaded image if prediction fails
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
""


