import tf = require("@tensorflow/tfjs-node");

import { BATCH_SIZE, CATEGORY_COUNT } from "../config";

import { TrainingData } from "../data";

export async function trainSequentialModel(data: TrainingData[]) {
  const model = getModel();
  await model.fit(
    tf.stack(data.map(value => value.activation)),
    tf.tensor1d(data.map(value => parseInt(value.category, 10))),
    { batchSize: BATCH_SIZE, epochs: 1, shuffle: true }
  );
  return model;
}

function getModel() {
  const model = tf.sequential();
  model.add(
    tf.layers.dense({ units: 250, activation: "relu", inputShape: [1024] })
  );
  model.add(tf.layers.dense({ units: 175, activation: "relu" }));
  model.add(tf.layers.dense({ units: 150, activation: "relu" }));
  model.add(tf.layers.dense({ units: CATEGORY_COUNT, activation: "softmax" }));

  model.compile({
    optimizer: tf.train.adam(),
    loss: "sparseCategoricalCrossentropy",
    metrics: ["accuracy"],
  });
  return model;
}
