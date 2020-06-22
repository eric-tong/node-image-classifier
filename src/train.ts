import { CATEGORIES } from "./config";
import { TrainingData } from "./data";

import tf = require("@tensorflow/tfjs-node");
import knn = require("@tensorflow-models/knn-classifier");

type ClassifierDataset = { [category: string]: tf.Tensor2D };

export async function trainKnnClassifier(data: TrainingData[]) {
  console.log("Training started");
  const classifier = knn.create();
  const dataset = getClassifierDataset(data);
  classifier.setClassifierDataset(dataset);
  return classifier;
}

function getClassifierDataset(data: TrainingData[]): ClassifierDataset {
  let result: ClassifierDataset = {};
  CATEGORIES.forEach((category) => {
    result[category] = tf.stack(
      data
        .filter((entry) => entry.category === category)
        .map((entry) => entry.activation)
    ) as tf.Tensor2D;
  });
  return result;
}
