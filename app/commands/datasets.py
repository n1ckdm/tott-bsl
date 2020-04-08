import typer
import time
from cv2 import cv2

from app.webcam import init_webcam, webcam_capture
from app.tf_ultils import save_data

app = typer.Typer()


@app.command("create")
def create_dataset(
    label: str,
    record_time: int = typer.Option(5, help="Seconds of data to record")
):
    """
    Records images from the webcam to generate a dataset for training
    """

    cam = init_webcam()
    timer = 4
    start_time = time.time()
    data = []
    while timer > -(record_time + 1):
        text = f"{timer}"
        if timer < 0:
            webcam_capture(cam, "Recording!!", data)
        else:
            webcam_capture(cam, text, None)

        if (time.time() - start_time) > 1:
            start_time = time.time()
            timer -= 1

    save_data(data, label)

    typer.echo(f"Recorded {len(data)} images for label: {label}")
    typer.echo("Exiting...")
    typer.Exit()
