import { getData, getArrayFromImage as getImageArrayFromPath } from "./data";

import { save } from "./utils/ModelUtils";

import tf = require("@tensorflow/tfjs-node");
import mobileNet = require("@tensorflow-models/mobilenet");
import knn = require("@tensorflow-models/knn-classifier");

train();

async function train() {
  const classifier = knn.create();
  const net = await mobileNet.load();
  const data = await getData("train");

  let index = 0;

  for (const { path, category } of data) {
    console.log(++index, data.length);

    const imageArray = await getImageArrayFromPath(path);
    const imageTensor = tf.node.decodePng(imageArray);
    const activation = net.infer(imageTensor, true);
    classifier.addExample(activation, category);
  }
  save(classifier);
}
