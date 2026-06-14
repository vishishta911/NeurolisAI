import cv2
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing import image

from services.predictor import model


LAST_CONV_LAYER = "Conv_1"


def generate_gradcam(
    image_path,
    save_path
):

    img = image.load_img(
        image_path,
        target_size=(224, 224)
    )

    img_array = image.img_to_array(img)

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    img_array = img_array / 255.0

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [
            model.get_layer(
                LAST_CONV_LAYER
            ).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(
            img_array
        )

        class_index = tf.argmax(
            predictions[0]
        )

        loss = predictions[
            :,
            class_index
        ]

    grads = tape.gradient(
        loss,
        conv_outputs
    )

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(
        pooled_grads * conv_outputs,
        axis=-1
    )

    heatmap = np.maximum(
        heatmap,
        0
    )

    heatmap /= np.max(
        heatmap
    )

    original = cv2.imread(
        image_path
    )

    heatmap = cv2.resize(
        heatmap,
        (
            original.shape[1],
            original.shape[0]
        )
    )

    heatmap = np.uint8(
        255 * heatmap
    )

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    superimposed = cv2.addWeighted(
        original,
        0.6,
        heatmap,
        0.4,
        0
    )

    cv2.imwrite(
        save_path,
        superimposed
    )

    return save_path