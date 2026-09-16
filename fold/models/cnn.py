"""The convolutional baseline.

Three 2-D convolutions over the (packet, feature) grid followed by a dense
classifier.  This is the model used in the conference version of this work,
kept here as the convolutional point of comparison: unlike the Tor-oriented
architectures, it was designed and evaluated on ODoH traces.
"""
from tensorflow import keras
from tensorflow.keras import layers


def build_cnn(sequence_length: int, num_features: int, num_classes: int) -> keras.Model:
    """Inputs ``cnn_input`` (batch, seq_len, num_features, 1); one softmax head."""
    cnn_input = keras.Input(shape=(sequence_length, num_features, 1), name="cnn_input")

    x = layers.Conv2D(32, (2, 2), activation="relu", padding="same", name="cnn_conv1")(cnn_input)
    x = layers.Conv2D(128, (2, 2), activation="relu", padding="same", name="cnn_conv2")(x)
    x = layers.Conv2D(256, (2, 2), activation="relu", padding="same", name="cnn_conv3")(x)
    x = layers.Flatten(name="cnn_flat")(x)
    x = layers.Dense(256, activation="relu", name="cnn_dense1")(x)
    x = layers.Dense(128, activation="relu", name="cnn_dense2")(x)
    x = layers.Dense(64, activation="relu", name="cnn_dense3")(x)

    x = layers.Dense(256, activation="relu", name="cls_dense1")(x)
    x = layers.BatchNormalization(name="cls_bn1")(x)
    x = layers.Dropout(0.3, name="cls_dropout1")(x)
    x = layers.Dense(128, activation="relu", name="cls_dense2")(x)
    x = layers.Dropout(0.2, name="cls_dropout2")(x)
    output = layers.Dense(num_classes, activation="softmax", name="multi_output")(x)

    return keras.Model(inputs={"cnn_input": cnn_input},
                       outputs={"multi_output": output},
                       name="cnn_attack_model")
