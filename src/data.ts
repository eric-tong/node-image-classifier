import { TRAINING_SIZE, VALIDATION_SIZE } from "./config";

import fs from "fs";
import neatCsv from "neat-csv";
import readline from "readline";
import { shuffle } from "./utils/ArrayUtils";

import tf = require("@tensorflow/tfjs-node");

export async function getData() {
  const data: { category: string; activation: tf.Tensor1D }[] = [];
  let completed = 0;

  await Promise.all(
    Array.from({ length: 42 }, async (_, i) => {
      const category = i.toString().padStart(2, "0");
      const activations = await getActivations(category);
      data.push(...activations);
      console.log("Getting data: ", `Completed ${++completed} of 42`);
    })
  );
  shuffle(data);

  return {
    training: data.slice(0, TRAINING_SIZE * data.length),
    validation: data.slice(data.length - VALIDATION_SIZE * data.length),
  };
}

async function getActivations(category: string) {
  const fileStream = fs.createReadStream(`.processed/${category}.csv`);
  const lines = readline.createInterface({
    input: fileStream,
    crlfDelay: Infinity,
  });
  const activations = [];

  for await (const line of lines) {
    const activation = line.split(",").map((str) => parseFloat(str));
    activations.push({ category, activation: tf.tensor1d(activation) });
  }
  return activations;
}

export async function getManifest(dataType: DataType) {
  const manifest = await getManifestFile(dataType);
  const data = manifest.map(({ filename, category }) => ({
    path:
      dataType === "train"
        ? `./dataset/train/train/${category}/${filename}`
        : `./dataset/test/test/${filename}`,
    category,
  }));
  return data;
}

async function getManifestFile(dataType: DataType) {
  const buffer = Buffer.from(
    await fs.promises.readFile(`./dataset/${dataType}.csv`)
  );
  const manifest = await neatCsv<ManifestEntry>(buffer);
  return manifest;
}

export async function getImageArrayFromPath(path: string) {
  const image = await fs.promises.readFile(path);
  const buffer = Buffer.from(image);
  return new Uint8Array(buffer);
}
