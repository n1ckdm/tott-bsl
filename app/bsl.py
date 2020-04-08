from app import app

from app.commands import datasets
from app.commands import model

app.add_typer(
    datasets.app,
    name="datasets",
    help="Record datasets for training"
)

app.add_typer(
    model.app,
    name="model",
    help="Run the model"
)

if __name__ == "__main__":
    app()