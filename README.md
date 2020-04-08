# Tech on The Tyne - Give me a sign!

## Pre-requisites:

If you're following along, we'll be doing this in VS Code on Windows. Note that you'll need Python 3.7 for Tensorflow to work. This is a base project structure just with a few folders setup to get us started.

Once you've cloned this git repository, `cd` into the root directory and create a new python virtual environment. The following instructions are for Windows:

```
<path-to-python37>\python.exe -m pip install --upgrade pip
<path-to-python37>\python.exe -m pip install --upgrade virtualenv
<path-to-python37>\python.exe -m virtualenv .venv
```

Now activate your virtual environment, in Windows that would look like this:

```
.\bsl-interp\Scripts\activate.ps1
```

You can check this worked by making sure that your `sys` prefix points to the virtual environment diretory just created:

```
python -c "import sys; print(sys.prefix)"
```

Now install all of the dependancies:

```
python -m pip install --upgrade -r .\requirements.txt
```

This will take a few minutes, make sure that there aren't any errors produced from the installation.

## Running the app

If everything has worked correctly you should be able to run the app:

```
python -m app.bsl --help
```

## Useful links

* [Tharsus](https://www.tharsus.co.uk/)
* [How to build a teachable machine](https://observablehq.com/@nsthorat/how-to-build-a-teachable-machine-with-tensorflow-js)
* [Teachable Machine with Google](https://teachablemachine.withgoogle.com/)
* [The Coding Train](https://thecodingtrain.com/)
* [Tensorflow](https://www.tensorflow.org/)
* [Scikit-Learn](https://scikit-learn.org/stable/)
* [Typer](https://typer.tiangolo.com/)
* [FastAPI](https://fastapi.tiangolo.com/)
* [BSL Fingerspelling Alphabet](https://www.british-sign.co.uk/fingerspelling-alphabet-charts/)