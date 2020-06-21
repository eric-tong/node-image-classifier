import fs from "fs";

import tf = require("@tensorflow/tfjs-node");
import knn = require("@tensorflow-models/knn-classifier");

type DatasetObject = [string, number[], [number, number]][];

export function save(classifier: knn.KNNClassifier) {
  const dataset = classifier.getClassifierDataset();
  const datasetObj: DatasetObject = Object.entries(
    dataset
  ).map(([label, data]) => [label, Array.from(data.dataSync()), data.shape]);
  const jsonStr = JSON.stringify(datasetObj);

  if (!fs.existsSync(".models")) {
    fs.mkdirSync(".models");
  }
  return fs.promises.writeFile(".models/knn.json", jsonStr);
}

export async function loadClassifier() {
  const buffer = await fs.promises.readFile(".models/knn.json");
  const datasetObj: DatasetObject = JSON.parse(buffer.toString());

  const tensorObj: { [label: string]: tf.Tensor2D } = datasetObj
    .map<[string, tf.Tensor2D]>(([label, data, shape]) => [
      label,
      tf.tensor(data, shape),
    ])
    .reduce((obj: { [label: string]: tf.Tensor2D }, [label, tensor]) => {
      obj[label] = tensor;
      return obj;
    }, {});

  const classifier = knn.create();
  classifier.setClassifierDataset(tensorObj);

  return classifier;
}
