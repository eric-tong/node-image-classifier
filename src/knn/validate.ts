import type knn from "@tensorflow-models/knn-classifier";
import { TrainingData } from "../data";
import tf = require("@tensorflow/tfjs-node");
import { PredictionResult } from "../analyze";

export async function validate(
  classifier: knn.KNNClassifier,
  data: TrainingData[]
) {
  console.log("Validation started");
  let completed = 0;

  const activations = Object.values(data).map(value => value.activation);
  const results: PredictionResult[] = [];

  for (const { category, activation } of Object.values(data)) {
    const result = await classifier.predictClass(activation);
    results.push({
      predicted: result.label,
      actual: category,
      confidences: result.confidences,
    });
    console.log(
      "Validation:",
      `Completed ${++completed} of ${activations.length}`
    );
  }

  return results;
}
