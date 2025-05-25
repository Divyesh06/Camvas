import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.layers import Dense,Flatten, Conv2D
from keras.layers import MaxPooling2D, Dropout
from keras.utils import to_categorical
import tensorflow as tf
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
import pickle
from keras.callbacks import TensorBoard
from tensorflow.keras.utils import plot_model

def keras_model(image_x, image_y):
    num_of_classes = 5
    model = Sequential([
    Flatten(input_shape=(28, 28, 1)),
    Dense(512, activation='relu'),
    Dense(512, activation='relu'),
    Dense(num_of_classes, activation='softmax')  
])

    model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)
    filepath = "ShapeDetection.keras"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    
    print(model.summary())
    return model, callbacks_list



def loadFromPickle():
    with open("features", "rb") as f:
        features = np.array(pickle.load(f))
    with open("labels", "rb") as f:
        labels = np.array(pickle.load(f))

    return features, labels


def augmentData(features, labels):
    features = np.append(features, features[:, :, ::-1], axis=0)
    labels = np.append(labels, -labels, axis=0)
    return features, labels


def prepress_labels(labels):
  #  labels = to_categorical(labels)
    return labels


features, labels = loadFromPickle()
# features, labels = augmentData(features, labels)
features, labels = shuffle(features, labels)
labels=prepress_labels(labels)
train_x, test_x, train_y, test_y = train_test_split(features, labels, random_state=0,
                                                    test_size=0.1)
train_x = train_x.reshape(train_x.shape[0], 28, 28, 1)
test_x = test_x.reshape(test_x.shape[0], 28, 28, 1)
model, callbacks_list = keras_model(28,28)
#plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
#print_summary(model)
model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs = 5, batch_size=64,
            callbacks=[TensorBoard(log_dir="ShapeDetection")])
model.save('ShapeDetection.keras')