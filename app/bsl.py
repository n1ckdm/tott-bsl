from app import app

@app.command("My first command")
def hello_world():
    print("hello from typer")

if __name__ == "__main__":
    app()