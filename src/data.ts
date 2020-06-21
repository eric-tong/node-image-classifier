import { TRAINING_SIZE, VALIDATION_SIZE } from "./config";

import fs from "fs";
import neatCsv from "neat-csv";

export async function getData(dataType: DataType) {
  const manifest = await getManifest(dataType);
  const data = manifest.map(({ filename, category }) => ({
    path:
      dataType === "train"
        ? `./dataset/train/train/${category}/${filename}`
        : `./dataset/test/test/${filename}`,
    category,
  }));
  return data;
}

async function getManifest(dataType: DataType) {
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
