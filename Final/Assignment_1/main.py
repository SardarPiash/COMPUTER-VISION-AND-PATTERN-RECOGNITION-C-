# from line 2-21-> Import the necessary libraries .....
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os

INIT_LR = 1e-4  # Initial learning rate
EPOCHS = 20   # Trainning epochs number
BS = 32  # Batch Size


DIRECTORY = r"C:\Desktop/Face_Mask_Project/dataset/"  # Indicates the path to the dataset directory
CATEGORIES = ["With Musk", "Without Face Musk"]  #Contains the labels for the images


# These two lines initialize empty lists to store the preprocessed image data in data = [] and corresponding labels in labels = []
data = []
labels = []
# These lines iterate on the catagories and images. Every image is converted into an array and preprocessed using the MobileNetV2 preprocessing function.
for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    for img in os.listdir(path):
    	img_path = os.path.join(path, img)
    	image = load_img(img_path, target_size=(224, 224))
    	image = img_to_array(image)
    	image = preprocess_input(image)

    	data.append(image)
    	labels.append(category)

# convert the text label into binary labels using LabelBinarizer..
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)  # by using it, make the labels one-hot
# Convert th list in to Numpy arrays which type is float32
data = np.array(data, dtype="float32")
labels = np.array(labels)

#The dataset is devided into training and testing sets using the train_test_split function.
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.20, stratify=labels, random_state=42)

# set the ImageDataGenerator object for data augmentation.
# aug object is used to perform real-time data augmentation on the training set.
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

# Initializing MobileNetV2 for transfer learning.
# 'weights' parameter is set to "imagenet" which means the pre-trained weights on the ImageNet dataset will be used.
# 'include_top' parameter is set to False to exclude fully connected layers.
# 'input_tensor' parameter specifies the shape of the input image.
baseModel = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))


headModel = baseModel.output  # 'headModel' variable represents the output tensor from the base model.
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)  # apply average pooling with a pool size of (7, 7) to reduce the spatial dimensions of the tensor.
headModel = Flatten(name="flatten")(headModel)  # Line flattens output tensor to 1D tensor.
headModel = Dense(128, activation="relu")(headModel)  # connected dense layer with 128 units and ReLU activation function.
headModel = Dropout(0.5)(headModel)  # Apply dropout regularisation at a rate of 0.5, where 50% of the input units are randomly set to 0 during training.
headModel = Dense(2, activation="softmax")(headModel)  # adds the final dense layer with 2 units and softmax activation function.

# creates the final model by specifying the inputs and outputs.
model = Model(inputs=baseModel.input, outputs=headModel)

# use for loop over each layer in the baseModel and set the trainable attribute to False.
for layer in baseModel.layers:
	layer.trainable = False

print("Compilation of the MODEL is going on...")
# create an instance of the Adam optimizer.
# 'lr' controls the step size taken during gradient descent updates.
# 'decay' reduces the learning rate over time.
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
# compile the model by specifying the loss function, optimizer, and evaluation metrics to be used during training.
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

print("Training Head Started")
# start the training process of the model using the fit function.
H = model.fit(
	aug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=EPOCHS)

print("Network evaluation")
#  use the trained model to make predictions on the test data.
predIdxs = model.predict(testX, batch_size=BS)

#
predIdxs = np.argmax(predIdxs, axis=1)

# print a classification report to evaluate the performance of the model's predictions.
print(classification_report(testY.argmax(axis=1), predIdxs,
	target_names=lb.classes_))


print("saving mask model")
#  model is saved with the specified filename "mask_detector.model".
model.save("mask_detector.model", save_format="h5")

# plot the training loss and accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")