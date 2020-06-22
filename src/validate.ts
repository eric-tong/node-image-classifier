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
      if (++completed % 100 === 0)
        console.log(
          "Validation:",
          `Completed ${completed} of ${activations.length}`
        );

      return result;
    });
  const results = await Promise.all(activations.map(predictClass));
  return results;
}
