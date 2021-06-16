# 3D face reconstruction
A Python programme using the Deep Learning librairies Pytorch and Pytorch3D that allows us to reconstruct any kind of face just by providing an image of someone.

## Installation

There is two options, first you use this code locally, second you use Google Colab. The second one will be comfortable to use once Pytorch3D is imported but his importation is long (20min). If you want to use it locally, it is **necessary** to have an efficient GPU.  

***If you run it locally :***
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the requirements to run python programmes.

```bash
pip install -r requirements.txt
```

In order to run our functions and models, you'll need to download these files : 
 1) [model2019_fullHead.h5](https://faces.dmi.unibas.ch/bfm/bfm2019.html)
 2) [shape_predictor_68_face_landmarks.dat](https://github.com/davisking/dlib-models/blob/master/shape_predictor_68_face_landmarks.dat.bz2)


## Results

If you're interested by the results it produces, you can just watch the (very) short video with the link below.

[![Watch the video](https://img.youtube.com/vi/h8PuR1Vn-RI/hqdefault.jpg)](https://youtu.be/h8PuR1Vn-RI)


## Usage

We provide a Jupiter Notebook that explains how to use it and tell every piece of information that will help to run it. If you feel interested, I strongly recommand to look directly at the notebook that you will find in the folder ***code*** as **Example_notebook.ipynb** and at the source code in python files.
