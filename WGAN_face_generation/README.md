### Project View


In this project, I implemented a WGAN for generating images of human faces, and the training images data sise >= 200K. The examples can be viewed in [link](https://github.com/tiandiao123/deep_learning/tree/master/face_generation/images_examples). For tutorial about WGAN, please check [here](https://lilianweng.github.io/lil-log/2017/08/20/from-GAN-to-WGAN.html)

### To Test my algorithm:

step1: download data set
```
python download_data.py
```
step2: train model:
```
python face_generation_wgan.py
```