import { BATCH_SIZE, CATEGORIES } from "../config";

import { PredictionResult } from "../analyze";
import { Tensor2D } from "@tensorflow/tfjs-node";
import { TrainingData } from "../data";

import tf = require("@tensorflow/tfjs-node");

export async function validate(model: tf.Sequential, data: TrainingData[]) {
  console.log("Validation started");
  let completed = 0;

  const activations = Object.values(data).map(value => value.activation);
  const results: PredictionResult[] = [];

  const predictions = model.predict(
    tf.stack(activations.slice(0, BATCH_SIZE))
  ) as Tensor2D;
  predictions.arraySync().forEach((prediction, i, arr) => {
    results.push({
      predicted: CATEGORIES[prediction.indexOf(Math.max(...prediction))],
      actual: data[i].category,
      confidences: arr.reduce(
        (obj, confidence, i) => ({ ...obj, [CATEGORIES[i]]: confidence }),
        {}
      ),
    });
  });

  console.log(
    "Validation:",
    `Completed ${++completed} of ${activations.length}`
  );

  return results;
}
