import { getData } from "./data";

import tf = require("@tensorflow/tfjs-node");
import mobileNet = require("@tensorflow-models/mobilenet");
import knn = require("@tensorflow-models/knn-classifier");

const classifier = knn.create();
getData("train").then(console.table);
// train();

async function train() {
  // const imageArray = await getArrayFromImage(
  //   "./dataset/train/train/03/3b97476019c26774236c618e4f936dbf.jpg"
  // );
  // const imageTensor = tf.node.decodePng(imageArray);
  // const net = await mobileNet.load();
  // const activation = net.infer(imageTensor, true);
  // console.log(activation);
  // classifier.addExample(activation, "03");
  // const result = await classifier.predictClass(activation);
  // console.log(result);
}
