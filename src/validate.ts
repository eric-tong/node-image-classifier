import { getData, getArrayFromImage as getImageArrayFromPath } from "./data";

import { loadClassifier } from "./utils/ModelUtils";
import { max } from "./utils/ArrayUtils";

import tf = require("@tensorflow/tfjs-node");
import mobileNet = require("@tensorflow-models/mobilenet");

validate();

async function validate() {
  const classifier = await loadClassifier();
  const net = await mobileNet.load();
  const data = await getData("train");

  let index = 0;

  for (const { path, category } of data) {
    console.log(++index, data.length);

    const imageArray = await getImageArrayFromPath(path);
    const imageTensor = tf.node.decodePng(imageArray);
    const activation = net.infer(imageTensor, true);
    const result = await classifier.predictClass(activation);
    console.log(max(result.confidences), category);
  }
}
