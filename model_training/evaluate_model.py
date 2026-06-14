import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

import numpy as np


model = tf.keras.models.load_model(
    "models/mobilenet_model.keras"
)


test_datagen = ImageDataGenerator(
    rescale=1./255
)


test_generator = test_datagen.flow_from_directory(
    "dataset/Testing",
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    shuffle=False
)


predictions = model.predict(
    test_generator
)

predicted_classes = np.argmax(
    predictions,
    axis=1
)

true_classes = test_generator.classes

class_labels = list(
    test_generator.class_indices.keys()
)

print(
    classification_report(
        true_classes,
        predicted_classes,
        target_names=class_labels
    )
)

print(
    confusion_matrix(
        true_classes,
        predicted_classes
    )
)