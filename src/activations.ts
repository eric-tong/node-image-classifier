import { getData, getImageArrayFromPath } from "./data";

import { TypedArray } from "@tensorflow/tfjs-node";
import fs from "fs";

import tf = require("@tensorflow/tfjs-node");
import mobileNet = require("@tensorflow-models/mobilenet");

findActivations();

async function findActivations() {
  const net = await mobileNet.load();
  const data = await getData("train");
  let completed = 0;

  createFiles(data);

  for (const { path, category } of data) {
    const imageArray = await getImageArrayFromPath(path);
    tf.tidy(() => {
      const imageTensor = tf.node.decodePng(imageArray);
      const activation = net.infer(imageTensor, true);
      saveActivation(activation.dataSync(), category);
    });
    console.log("Activations:", `Processed ${++completed} of ${data.length}`);
  }
}

function createFiles(array: { category: string }[]) {
  if (!fs.existsSync(".processed")) {
    fs.mkdirSync(".processed");
  }
  const categories = new Set(array.map(({ category }) => category));
  categories.forEach((category) =>
    fs.writeFileSync(`.processed/${category}.csv`, "")
  );
}

function saveActivation(activation: TypedArray, category: string) {
  fs.appendFileSync(`.processed/${category}.csv`, activation.join(","));
  fs.appendFileSync(`.processed/${category}.csv`, "\n");
}
