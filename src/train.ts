import { getData, getArrayFromImage as getImageArrayFromPath } from "./data";
import { getRandomSample, max } from "./utils";

import tf = require("@tensorflow/tfjs-node");
import mobileNet = require("@tensorflow-models/mobilenet");
import knn = require("@tensorflow-models/knn-classifier");

const SAMPLE_SIZE = 100;

train().then(test);

async function train() {
  const classifier = knn.create();
  const net = await mobileNet.load();
  const data = await getData("train");
  const sample = getRandomSample(data, SAMPLE_SIZE);

  let index = 0;

  for (const { path, category } of sample) {
    console.log(++index, sample.length);

    const imageArray = await getImageArrayFromPath(path);
    const imageTensor = tf.node.decodePng(imageArray);
    const activation = net.infer(imageTensor, true);
    classifier.addExample(activation, category);
  }
  return classifier;
}

async function test(classifier: knn.KNNClassifier) {
  const net = await mobileNet.load();
  const data = await getData("train");
  const sample = getRandomSample(data, SAMPLE_SIZE);

  for (const { path, category } of sample) {
    const imageArray = await getImageArrayFromPath(path);
    const imageTensor = tf.node.decodePng(imageArray);
    const activation = net.infer(imageTensor, true);
    const result = await classifier.predictClass(activation);
    console.log(max(result.confidences), category);
  }
}
