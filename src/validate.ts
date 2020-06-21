import { max } from "./utils/ArrayUtils";
import type knn from "@tensorflow-models/knn-classifier";
import type tf from "@tensorflow/tfjs-node";

export async function validate(
  classifier: knn.KNNClassifier,
  data: { category: string; activation: tf.Tensor1D }[]
) {
  let completed = 0;
  let correct = 0;

  for (let index = 0; index < data.length; index++) {
    const { activation, category } = data[index];
    const result = await classifier.predictClass(activation);

    if (max(result.confidences) === category) {
      correct++;
    }
    console.log(
      "Validation:",
      `Processed ${++completed} of ${data.length}`,
      `Accuracy: ${correct / index}`
    );
  }

  return "Complete";
}
