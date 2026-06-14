import numpy as np

import tensorflow as tf

from tensorflow.keras.preprocessing import image


MODEL_PATH = "models/mobilenet_model.keras"

model = tf.keras.models.load_model(
    MODEL_PATH
)


CLASS_NAMES = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary"
]


def predict_mri(image_path):

    img = image.load_img(
        image_path,
        target_size=(224, 224)
    )

    img_array = image.img_to_array(img)

    img_array = img_array / 255.0

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    predictions = model.predict(
        img_array,
        verbose=0
    )

    predicted_index = np.argmax(
        predictions
    )

    confidence = float(
        np.max(predictions)
    ) * 100

    prediction = CLASS_NAMES[
        predicted_index
    ]

    return prediction, confidence, predictions[0]