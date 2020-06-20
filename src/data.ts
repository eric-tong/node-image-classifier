import fs from "fs";
import { getRandomSample } from "./utils";
import neatCsv from "neat-csv";
import { promisify } from "util";

const readFile = promisify(fs.readFile);

export async function getData(dataType: DataType) {
  const manifest = await getManifest(dataType);
  const sample = getRandomSample(manifest, 100);
  return sample.map(({ filename, category }) => ({
    path:
      dataType === "train"
        ? `./dataset/train/train/${category}/${filename}`
        : `./dataset/test/test/${filename}`,
    category,
  }));
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
