import { getData, getImageArrayFromPath } from "./data";

import { save } from "./utils/ModelUtils";

import tf = require("@tensorflow/tfjs-node");
import mobileNet = require("@tensorflow-models/mobilenet");
import knn = require("@tensorflow-models/knn-classifier");

train();

async function train() {
  const classifier = knn.create();
  const net = await mobileNet.load();
  const data = await getData("train");
  let completed = 0;

  for (const { path, category } of data) {
    const imageArray = await getImageArrayFromPath(path);
    const imageTensor = tf.node.decodePng(imageArray);
    const activation = net.infer(imageTensor, true);
    classifier.addExample(activation, category);
    console.log("Training:", `Processed ${++completed} of ${data.length}`);
  }
  save(classifier);
}
