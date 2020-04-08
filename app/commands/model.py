from datetime import datetime
import typer

import numpy as np
import tensorflow as tf

from cv2 import cv2

from app.tf_ultils import load_classifier, load_label_dict, process_image, init_model
from app.webcam import init_webcam, web_imgs, add_text_to_frame

app = typer.Typer()


def get_label(labels, num):
    return list(labels.keys())[list(labels.values()).index(num)]


@app.command("run")
def run_model():
    """
    Runs the previously trained model
    """

    classifier = load_classifier()
    labels = load_label_dict()
    model = init_model()
    camera = init_webcam()

    for frame in web_imgs(camera):
        img = process_image(frame)
        prediction = model.predict(img[np.newaxis, ...])
        probs = tf.nn.softmax(prediction).numpy()
        res = classifier.predict(probs)[0]
        prob = classifier.predict_proba(probs)[0]

        label = get_label(labels, res)
        add_text_to_frame(frame, f"{label} conf: {prob}")

        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    typer.echo("Exiting")
    typer.Exit()
