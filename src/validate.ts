import { getData, getImageArrayFromPath } from "./data";

import { loadClassifier } from "./utils/ModelUtils";
import { max } from "./utils/ArrayUtils";

import tf = require("@tensorflow/tfjs-node");
import mobileNet = require("@tensorflow-models/mobilenet");

validate();

async function validate() {
  const classifier = await loadClassifier();
  const net = await mobileNet.load();
  const data = await getData("train");

  let completed = 0;
  let correct = 0;

  for (let index = 0; index < data.length; index++) {
    const { path, category } = data[index];
    const imageArray = await getImageArrayFromPath(path);
    const imageTensor = tf.node.decodePng(imageArray);
    const activation = net.infer(imageTensor, true);
    const result = await classifier.predictClass(activation);

    if (max(result.confidences) === category) {
      correct++;
    }
    console.log(
      "Validation:",
      `Processed ${++completed} of ${data.length}`,
      `Accuracy: ${correct / index}`
    );
  }
}
