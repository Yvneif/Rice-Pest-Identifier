Web
pip install Flask

from flask import Flask, render_template, request
import tensorflow as tf  # Import for loading the pre-trained model

app = Flask(__name__)

# Load your pre-trained model (replace with your model loading logic)
model = tf.keras.models.load_model('path/to/your/model.h5')  # Assuming you saved the model

@app.route('/')
def upload_form():
    return render_template('index.html')  # Renders an HTML page with an upload form

@app.route('/', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the uploaded image file
        image_file = request.files['image']

        # Preprocess the image (similar to your code)
        # ... (resize, convert to array, normalize)

        # Make prediction using your model
        prediction = model.predict(np.expand_dims(image_array, axis=0))
        predicted_class = np.argmax(prediction)

        # Render the prediction result in the template
        return render_template('index.html', prediction=predicted_class)

if __name__ == '__main__':
    app.run(debug=True)

