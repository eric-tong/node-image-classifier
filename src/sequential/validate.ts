import { BATCH_SIZE, CATEGORIES } from "../config";
import { PredictionResult, analyze } from "../analyze";
import { TrainingData, getData } from "../data";

import { Tensor2D } from "@tensorflow/tfjs-node";

import tf = require("@tensorflow/tfjs-node");

getData()
  .then(({ validation }) => validate(validation))
  .then(analyze)
  .then(console.table);

export async function validate(data: TrainingData[]) {
  console.log("Validation started");

  const model = await tf.loadLayersModel(
    `file://${__dirname}/.models/model.json`
  );
  const activations = Object.values(data).map(value => value.activation);
  const zeros = tf.tensor1d(Array.from({ length: 1024 }, () => 0));
  const results: PredictionResult[] = [];

  for (let i = 0; i < activations.length; i += 64) {
    let batch;
    if (i + 64 <= activations.length) {
      batch = tf.stack(activations.slice(i, i + 64));
    } else {
      const padding = Array.from(
        { length: 64 - (activations.length % 64) },
        () => zeros
      );
      batch = tf.stack([
        ...activations.slice(i, activations.length),
        ...padding,
      ]);
    }

    const predictions = model.predict(batch) as Tensor2D;
    predictions.arraySync().forEach((prediction, k) => {
      if (i + k < data.length)
        results.push({
          predicted: CATEGORIES[prediction.indexOf(Math.max(...prediction))],
          actual: data[i + k].category,
          confidences: prediction.reduce(
            (obj, confidence, j) => ({ ...obj, [CATEGORIES[j]]: confidence }),
            {}
          ),
        });
    });

    console.log("Validation:", `Completed ${i} of ${activations.length}`);
  }

  return results;
}
