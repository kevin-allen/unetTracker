# Running unetTracker on Google colab

The GUI components of unetTracker are currently not working in Google Colab. We hope to have a solution for this soon.

The problem is described here:
https://stackoverflow.com/questions/76492248/how-can-i-display-a-matplotlib-fig-canvas-inside-a-ipywidgets-vbox-in-google-col

## Install unetTracker on Google colab

```
!git clone https://github.com/kevin-allen/unetTracker
!pip install -r unetTracker/requirements.txt
!pip install -e unetTracker
```
