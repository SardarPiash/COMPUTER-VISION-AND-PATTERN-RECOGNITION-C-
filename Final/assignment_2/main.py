from google.colab import drive
drive.mount('/content/drive')

# Import the required libraries
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16

# Load the pre-trained VGG16 model
base_model = VGG16(include_top=False, input_shape=(224, 224, 3))
# Freeze the layers of the VGG16 model
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers to the VGG16 model
x = base_model.output
x = Flatten()(x)
x = Dense(1024, activation='relu')(x)
x = Dense(512, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)
#model.add(Dense(10, activation='softmax'))

# Create the new model with the VGG16 base and custom layers
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Print the model summary
model.summary()

# Define the image preprocessing function
def preprocess_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.keras.applications.vgg16.preprocess_input(img)
    return img

# Define the object detection function
def detect_objects(image_path):
    import cv2
    import numpy as np

model.save('/content/drive/My Drive/obj_det/Kaium_Object_detection_model.h5')

# Load the model
model = tf.keras.models.load_model('/content/drive/My Drive/obj_det/Kaium_Object_detection_model.h5')

image_path = '/content/drive/My Drive/obj_det/car1.jpg'

# Load the image
image = cv2.imread(image_path)

# Preprocess the image
image = preprocess_image(image_path)

# Run the model to detect objects
predictions = model.predict(np.array([image]))

# Get the class with the highest probability
class_idx = np.argmax(predictions[0])

# Print the predicted class
if class_idx == 0:
    print('This is a cat')
else:
    print('This is a car')