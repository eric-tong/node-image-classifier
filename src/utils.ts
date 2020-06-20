export function getRandomSample<T>(array: T[], size: number) {
  return shuffle([...array]).slice(0, size);
}

function shuffle<T>(array: T[]) {
  var currentIndex = array.length,
    temporaryValue,
    randomIndex;

  while (0 !== currentIndex) {
    randomIndex = Math.floor(Math.random() * currentIndex);
    currentIndex -= 1;

    temporaryValue = array[currentIndex];
    array[currentIndex] = array[randomIndex];
    array[randomIndex] = temporaryValue;
  }

  return array;
}

export function max(array: { [category: string]: number }) {
  let maxCategory = "";
  let maxProbablity = 0;

  for (const [category, probability] of Object.entries(array)) {
    if (probability > maxProbablity) maxCategory = category;
  }
  return maxCategory;
}
