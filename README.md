# Simple Image Classifier with Node.js

Simple image classifier using tensorflow.js and MobileNet

## Getting started

Clone this repository

```
git clone https://github.com/eric-tong/node-image-classifier.git <project_name>
```

Change into the new project directory and install the dependencies

```
yarn install
```

Start the compilation watcher

```
yarn compile
```

Preprocess by using MobileNet to find activations

```
yarn preprocess
```

Finally, use Sequential or KNN models to predict classes

```
(Sequential)
yarn train
yarn validate

(KNN)
yarn knn
```
