from flask import Flask, request, jsonify
from keras.preprocessing import image
from keras.models import load_model
import pickle
import numpy as np
from PIL import Image
from flask import Flask
app = Flask(__name__)

from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import os

app = Flask(__name__)

# Define a directory to save uploaded images
UPLOAD_FOLDER = 'uploads_for_prediction'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Sample predict function (replace with actual prediction logic)
def predict(image_path):
    # Placeholder: Load the image and perform the actual prediction
    # For example, return a mock result
    model = load_model('path_to_save_model/cnn_model.h5')
   # with open('lungcancer.pkl','rb') as file:
  #      model = pickle.load(file)
    
   # return model.predict(image_path)
    target_size=(128,128)
    img = image.load_img(image_path, target_size=target_size)  # Load image
    img_array = image.img_to_array(img)  # Convert to array
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions for batch
    img_array /= 255.0  # Normalize to [0, 1]
    predictions = model.predict(img_array)  # Get predictions
    predicted_class = np.argmax(predictions, axis=1)  # Get the index of the highest prediction
    confidence = np.max(predictions)  # Get the highest
    return {"predicted_class": int(predicted_class), "confidence": float(confidence)}
    #return image_path


# Route to upload image and call prediction
@app.route('/', methods=['GET', 'POST'])
def upload_and_predict():
    if request.method == 'POST':
        if 'image' not in request.files:
            return jsonify({"error": "No image part in the request"}), 400
        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        if file:
            # Save the uploaded image to the upload folder
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Call the predict function
            prediction = predict(file_path)
            
            # Return the prediction result as JSON
            return jsonify(prediction)
    
    # If GET request, display an upload form
    return '''
    <!doctype html>
    <title>Upload an Image for Prediction</title>
    <h1>Upload an Image for Prediction</h1>
    <form method="post" enctype="multipart/form-data">
        <input type="file" name="image">
        <input type="submit" value="Upload and Predict">
    </form>
    '''

if __name__ == '__main__':
    app.run(debug=True)
