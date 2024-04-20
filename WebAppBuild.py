import streamlit as st
import tensorflow as tf  # Import for loading the pre-trained model

# Load your pre-trained model (replace with your model loading logic)
model = tf.keras.models.load_model('path/to/your/model.h5')  # Assuming you saved the model

st.title('Image Identification App')

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = tf.keras.preprocessing.image.load_img(uploaded_file, target_size=(224, 224))
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    prediction = model.predict(image_array)
    predicted_class = np.argmax(prediction)

    st.write("Predicted Class:", predicted_class)
