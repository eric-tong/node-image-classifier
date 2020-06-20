import fs from "fs";
import neatCsv from "neat-csv";
import { promisify } from "util";

import tf = require("@tensorflow/tfjs-node");
import mobileNet = require("@tensorflow-models/mobilenet");
import knn = require("@tensorflow-models/knn-classifier");

console.log("Version", tf.version.tfjs);
const readFile = promisify(fs.readFile);
const classifier = knn.create();
main();
// getTrainingData();

async function main() {
  const imageArray = await getArrayFromImage(
    "./dataset/train/train/03/3b97476019c26774236c618e4f936dbf.jpg"
  );
  const imageTensor = tf.node.decodePng(imageArray);

  const net = await mobileNet.load();
  const activation = net.infer(imageTensor, true);
  console.log(activation);

  classifier.addExample(activation, "03");

  const result = await classifier.predictClass(activation);
  console.log(result);
}

async function getTrainingData() {
  const buffer = Buffer.from(await readFile("./dataset/train.csv"));
  const data = await neatCsv(buffer);
  console.log(data);
}

async function getArrayFromImage(path: string) {
  const image = await readFile(path);
  const buffer = Buffer.from(image);
  return new Uint8Array(buffer);
}
