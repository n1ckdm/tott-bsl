import typer
from typing import List
from cv2 import cv2


def init_webcam() -> cv2.VideoCapture:
    typer.echo("Starting webcam...")
    return cv2.VideoCapture(1)


def add_text_to_frame(frame: cv2.VideoCapture, text: str):
    bottomLeftCornerOfText = (10, 40)
    fontScale = 1
    fontColor = (50,50,50)
    lineType = 2

    cv2.putText(
        frame,
        text,
        bottomLeftCornerOfText,
        cv2.FONT_HERSHEY_SIMPLEX,
        fontScale,
        fontColor,
        lineType
    )


def webcam_capture(
    camera: cv2.VideoCapture,
    text: str = None,
    data: List[cv2.VideoCapture] = None
):
    
    _, frame = camera.read()

    if text is not None:
        add_text_to_frame(frame, text)

    cv2.imshow("Sign language interpretor", frame)
    cv2.waitKey(1)

    if data is not None:
        data.append(frame)


def web_imgs(cam: cv2.VideoCapture):

    while True:
        _, frame = cam.read()
        yield frame
