from flask import Flask, request, render_template
import numpy as np
from PIL import Image
from io import BytesIO
import tensorflow as tf

app = Flask(__name__)

MODEL = tf.keras.models.load_model("saved_models/1")
CLASS_NAMES = ['Apple___Black_rot',
 'Apple___healthy',
 'Cherry_(including_sour)___Powdery_mildew',
 'Cherry_(including_sour)___healthy',
 'Corn_(maize)___Common_rust_',
 'Corn_(maize)___healthy',
 'Grape___Esca_(Black_Measles)',
 'Grape___healthy',
 'Peach___Bacterial_spot',
 'Peach___healthy',
 'Pepper,_bell___Bacterial_spot',
 'Pepper,_bell___healthy',
 'Potato___Late_blight',
 'Potato___healthy',
 'Strawberry___Leaf_scorch',
 'Strawberry___healthy',
 'Tomato___Late_blight',
 'Tomato___healthy']

@app.route("/")
def home():
    return render_template("index.html")

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.route("/about_us")
def about_us():
    return render_template("about_us.html")

@app.route("/predict", methods=["POST"])
def predict():
    image = read_file_as_image(request.files["file"].read())
    img_batch = np.expand_dims(image, 0)
    
    predictions = MODEL.predict(img_batch)
    
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0])) * 100
    
    return render_template("index.html", prediction=predicted_class, confidence=confidence)


if __name__ == "__main__":
    app.run(debug=True)
