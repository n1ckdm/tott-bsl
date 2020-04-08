import typer
from typing import List, Dict
from cv2 import cv2
import pickle
import os

import numpy as np
import tensorflow as tf
from sklearn import neighbors

IMAGE_SIZE = 128
LABEL_FILE = 'labels.pkl'
CLASS_FILE = 'class.pkl'
LOAGITS_FILE = 'logits.npy'
LABELS_FILE = 'labels.npy'
WEIGHTS = "distance"
NN = 6

def init_model():
    IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SHAPE,
        include_top=False,
        weights='imagenet'
    )
    base_model.trinable = False
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    model = tf.keras.Sequential([
        base_model,
        global_average_layer
    ])
    typer.echo("Model finished loading")
    return model


def process_image(img: cv2.VideoCapture):
    img = (tf.cast(img, tf.float32)/127.5) -1
    img = tf.image.resize(img, (IMAGE_SIZE, IMAGE_SIZE)).numpy()
    return img


def load_label_dict() -> Dict[str, int]:
    if os.path.isfile(LABEL_FILE):
        with open(LABEL_FILE, "rb") as f:
            return pickle.loads(f.read())
    else:
        return {}


def save_label_dict(label_dict: Dict[str, int]):
    with open(LABEL_FILE, "wb+") as f:
        f.write(pickle.dumps(label_dict))


def set_label_number(label: str, label_dict: Dict[str, int]) -> int:
    if label in label_dict:
        return label_dict[label]
    else:
        num = len(label_dict)
        label_dict[label] = num
        return num


def load_image_data(
    prev_logits: np.ndarray,
    prev_labels: np.ndarray,
):
    if os.path.isfile(LOAGITS_FILE) and os.path.isfile(LABELS_FILE):
        return (
            np.concatenate([prev_logits, np.load(LOAGITS_FILE)]),
            np.concatenate([prev_labels, np.load(LABELS_FILE)])
        )
    else:
        return prev_logits, prev_labels


def save_image_data(
    logits: np.ndarray,
    labels: np.ndarray,
):
    if os.path.isfile(LOAGITS_FILE):
        os.remove(LOAGITS_FILE)
    np.save(LOAGITS_FILE, logits)

    if os.path.isfile(LABELS_FILE):
        os.remove(LABELS_FILE)
    np.save(LABELS_FILE, labels)


def load_classifier() -> neighbors.KNeighborsClassifier:
    if os.path.isfile(CLASS_FILE):
        with open(CLASS_FILE, "rb") as f:
            return pickle.loads(f.read())
    else:
        return neighbors.KNeighborsClassifier(NN, weights=WEIGHTS)


def save_classifier(knn: neighbors.KNeighborsClassifier):
    with open(CLASS_FILE, "wb+") as f:
        f.write(pickle.dumps(knn))


def save_data(
    data: List[cv2.VideoCapture],
    label: str
):
    # load the base ImageNet model
    model = init_model()

    conv_imgs = []
    for img in data:
        conv_imgs.append(process_image(img))

    images4d = np.asarray(conv_imgs)
    predictions = model.predict(images4d)
    probs = tf.nn.softmax(predictions).numpy()

    # Load our label data
    label_dict = load_label_dict()
    label_num = set_label_number(label, label_dict)
    labels = [label_num] * len(conv_imgs)

    # Load in our previous model data
    probs, labels = load_image_data(probs, labels)

    # Update our model
    classifier = load_classifier()
    classifier.fit(probs, labels)

    # Save the model and data
    save_label_dict(label_dict)
    save_classifier(classifier)
    save_image_data(probs, labels)







