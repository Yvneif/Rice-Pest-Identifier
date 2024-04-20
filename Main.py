Main 

import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Define data paths (modify these)
data_dir = 'path/to/your/dataset'
train_dir = os.path.join(data_dir, 'train')  # Folder for training data
val_dir = os.path.join(data_dir, 'val')  # Optional folder for validation data (if using)

# Set image dimensions (adjust if needed)
img_height, img_width = 224, 224

# Load data (using manual loading for this example)
train_images, train_labels = [], []
for class_dir in os.listdir(train_dir):
  class_path = os.path.join(train_dir, class_dir)
  for img_name in os.listdir(class_path):
    img = load_img(os.path.join(class_path, img_name), target_size=(img_height, img_width))
    img_array = img_to_array(img) / 255  # Normalize pixel values
    train_images.append(img_array)
    train_labels.append(class_dir)  # Label based on directory name (modify if needed)

train_images = np.array(train_images)
train_labels = np.array(train_labels)

# Validation data loading (optional, modify similarly to training data loading)
val_images, val_labels = [], []
# ... (load validation data if using separate folder)

# Data augmentation (optional, for training data)
datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
datagen.fit(train_images)

# Define the model (replace/adjust layers as needed)
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
# ... add more convolutional layers
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(len(os.listdir(train_dir)), activation='softmax'))  # Adjust for number of classes

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(datagen.flow(train_images, train_labels, batch_size=32), epochs=10,
          validation_data=(val_images, val_labels) if val_images else None)

# Evaluate the model (optional)
# loss, accuracy = model.evaluate(test_images, test_labels)
# print('Accuracy:', accuracy)

# Use the model to predict a new image (modify loading and preprocessing)
new_image_path = 'path/to/your/new/image.jpg'
new_image = load_img(new_image_path, target_size=(img_height, img_width))
new_image_array = img_to_array(new_image) / 255
new_image_array = np.expand_dims(new_image_array, axis=0)  # Reshape for prediction
prediction = model.predict(new_image_array)
predicted_class = np.argmax(prediction)  # Get index of most likely class

# Print the predicted class (assuming class names match folder names)
print("Predicted class:", predicted_class)
