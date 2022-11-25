import setuptools
from setuptools.extension import Extension

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setuptools.setup(
    name="unet-tracker", # Replace with your own username
    version="0.0.1",
    author="Kevin Allen",
    author_email="allen@uni-heidelberg.de",
    description="A python package to train U-Net to track objects in videos",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kevin-allen/unet-tracker",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Operating System :: POSIX :: Linux",
    ],
    python_requires='>=3.6'
)
