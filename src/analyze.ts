import { max } from "./utils/ArrayUtils";

type ProcessedData = {
  actual: string;
  predicted: string;
  confidences: { category: string; confidence: number }[];
};

export function analyze(
  results: {
    label: string;
    confidences: { [label: string]: number };
  }[]
) {
  const data: ProcessedData[] = results.map(result => ({
    actual: result.label,
    predicted: max(result.confidences),
    confidences: Object.entries(result.confidences)
      .map(([category, confidence]) => ({ category, confidence }))
      .sort((a, b) => b.confidence - a.confidence),
  }));

  return {
    top1: top1Accuracy(data),
    top5: top5Accuracy(data),
  };
}

function top1Accuracy(data: ProcessedData[]) {
  return (
    data.filter(value => value.actual === value.predicted).length / data.length
  );
}

function top5Accuracy(data: ProcessedData[]) {
  return (
    data.filter(value =>
      value.confidences
        .slice(0, 5)
        .map(confidence => confidence.category)
        .includes(value.actual)
    ).length / data.length
  );
}
