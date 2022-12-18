FROM nvcr.io/nvidia/pytorch:22.11-py3

WORKDIR /usr/src/app

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir ipywidgets ipympl albumentations


CMD ["jupyter-lab","--ip=0.0.0.0","--no-browser","--allow-root", "--NotebookApp.token=unet"]
