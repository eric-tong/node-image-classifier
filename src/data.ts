import { TRAINING_SIZE, VALIDATION_SIZE } from "./config";

import fs from "fs";
import neatCsv from "neat-csv";
import { promisify } from "util";
import { shuffle } from "./utils/ArrayUtils";

const readFile = promisify(fs.readFile);

export async function getData(dataType: DataType) {
  const manifest = await getManifest(dataType);
  const data = manifest.map(({ filename, category }) => ({
    path:
      dataType === "train"
        ? `./dataset/train/train/${category}/${filename}`
        : `./dataset/test/test/${filename}`,
    category,
  }));

  shuffle(data);
  return dataType === "train"
    ? data.slice(0, TRAINING_SIZE)
    : data.slice(data.length - VALIDATION_SIZE);
}

async function getManifest(dataType: DataType) {
  const buffer = Buffer.from(await readFile(`./dataset/${dataType}.csv`));
  const manifest = await neatCsv<ManifestEntry>(buffer);
  return manifest;
}

export async function getArrayFromImage(path: string) {
  const image = await readFile(path);
  const buffer = Buffer.from(image);
  return new Uint8Array(buffer);
}
