type ProcessedData = {
  actual: string;
  predicted: string;
  confidences: { category: string; confidence: number }[];
};

export function analyze(
  results: {
    predicted: string;
    actual: string;
    confidences: { [label: string]: number };
  }[]
) {
  const data: ProcessedData[] = results.map(result => ({
    actual: result.predicted,
    predicted: result.actual,
    confidences: Object.entries(result.confidences)
      .map(([category, confidence]) => ({ category, confidence }))
      .sort((a, b) => b.confidence - a.confidence),
  }));

  return {
    top1: top1Accuracy(data),
    top3: top3Accuracy(data),
    ...topMisclassifications(data),
  };
}

function top1Accuracy(data: ProcessedData[]) {
  return (
    data.filter(value => value.actual === value.predicted).length / data.length
  );
}

function top3Accuracy(data: ProcessedData[]) {
  return (
    data.filter(value =>
      value.confidences
        .slice(0, 3)
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
