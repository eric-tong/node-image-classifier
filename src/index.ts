import fs from "fs";
import { promisify } from "util";

import tf = require("@tensorflow/tfjs-node");
import mobileNet = require("@tensorflow-models/mobilenet");

console.log("Version", tf.version.tfjs);
const readFile = promisify(fs.readFile);
main();

async function main() {
  const imageArray = await getArrayFromImage("./dataset/cat.jpg");
  const imageTensor = tf.node.decodePng(imageArray);

  const net = await mobileNet.load();
  const result = await net.classify(imageTensor);
  console.log(result);
}

async function getArrayFromImage(path: string) {
  const image = await readFile(path);
  const buf = Buffer.from(image);
  return new Uint8Array(buf);
}
