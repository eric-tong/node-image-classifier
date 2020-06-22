export function shuffle<T>(array: T[]) {
  const generate = getRandomGenerator();
  let currentIndex = array.length,
    temporaryValue,
    randomIndex;

  while (0 !== currentIndex) {
    randomIndex = Math.floor(generate() * currentIndex);
    currentIndex -= 1;

    temporaryValue = array[currentIndex];
    array[currentIndex] = array[randomIndex];
    array[randomIndex] = temporaryValue;
  }

  return array;
}

function getRandomGenerator() {
  let seed = 0;
  return () => Math.sin(++seed) / 2 + 0.5;
}
