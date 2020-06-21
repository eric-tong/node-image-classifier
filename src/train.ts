import knn = require("@tensorflow-models/knn-classifier");
import type tf from "@tensorflow/tfjs-node";

export async function trainKnnClassifier(
  data: { category: string; activation: tf.Tensor1D }[]
) {
  const classifier = knn.create();
  let complete = 0;

  for (const { activation, category } of data) {
    classifier.addExample(activation, category);
    console.log("Training: ", `Completed ${complete} of ${data.length}`);
  }

  return classifier;
}
