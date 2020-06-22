export type PredictionResult = {
  predicted: string;
  actual: string;
  confidences: { [label: string]: number };
};

type ProcessedData = {
  actual: string;
  predicted: string;
  confidences: { category: string; confidence: number }[];
};

export function analyze(results: PredictionResult[]) {
  const data: ProcessedData[] = results.map(result => ({
    actual: result.actual,
    predicted: result.predicted,
    confidences: Object.entries(result.confidences)
      .map(([category, confidence]) => ({ category, confidence }))
      .sort((a, b) => b.confidence - a.confidence),
  }));

  console.table(
    misclassifiedConfidences(data.map(val => ({ ...val, ...val.confidences })))
  );
  return {
    top1: topKAccuracy(data, 1),
    top2: topKAccuracy(data, 2),
    top3: topKAccuracy(data, 3),
    top4: topKAccuracy(data, 4),
    top5: topKAccuracy(data, 5),
    ...topMisclassifications(data),
  };
}

function topKAccuracy(data: ProcessedData[], k: number) {
  return (
    data.filter(value =>
      value.confidences
        .slice(0, k)
        .map(confidence => confidence.category)
        .includes(value.actual)
    ).length / data.length
  );
}

function topMisclassifications(data: ProcessedData[]) {
  const counts = new Map<string, number>();
  data.forEach(value => {
    if (value.actual !== value.predicted) {
      const misclassification = `${value.actual} ${value.predicted}`;
      const count = counts.get(misclassification) ?? 0;
      counts.set(misclassification, count + 1);
    }
  });

  const result: { [label: string]: number } = {};
  Array.from(counts.entries())
    .sort((a, b) => b[1] - a[1])
    .slice(0, counts.size / 2)
    .forEach(val => {
      result[val[0]] = val[1];
    });
  return result;
}

function misclassifiedConfidences(data: ProcessedData[]) {
  return data.filter(val => val.actual !== val.predicted);
}
