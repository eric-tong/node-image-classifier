import { max } from "./utils/ArrayUtils";
import type knn from "@tensorflow-models/knn-classifier";
import { TrainingData } from "./data";

import tf = require("@tensorflow/tfjs-node");

export async function validate(
  classifier: knn.KNNClassifier,
  data: TrainingData[]
) {
  console.log("Validation started");
  let completed = 0;

  const activations = Object.values(data).map(value => value.activation);
  const predictClass = (activation: tf.Tensor1D) =>
    classifier.predictClass(activation).then(result => {
      console.log(
        "Validation:",
        `Completed ${++completed} of ${activations.length}`
      );

      const predicted = max(result.confidences);
      const actual = result.label;
      return {
        correct: actual === predicted,
        actual,
        predicted,
        ...result.confidences,
      };
    });
  const results = await Promise.all(activations.map(predictClass));
  console.table(results);

  return "Complete";
}
