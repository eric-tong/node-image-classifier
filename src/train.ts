import { getData, getArrayFromImage as getImageArrayFromPath } from "./data";

import tf = require("@tensorflow/tfjs-node");
import mobileNet = require("@tensorflow-models/mobilenet");
import knn = require("@tensorflow-models/knn-classifier");

train();

async function train() {
  const classifier = knn.create();
  const net = await mobileNet.load();
  const data = await getData("train");

  for (const { path, category } of data) {
    const imageArray = await getImageArrayFromPath(path);
    const imageTensor = tf.node.decodePng(imageArray);
    const activation = net.infer(imageTensor, true);
    console.log(activation);
    classifier.addExample(activation, category);

    const result = await classifier.predictClass(activation);
    console.log(result, category);
  }
}
