export const TRAINING_SIZE = 0.2;
export const VALIDATION_SIZE = 0.01;
export const CATEGORY_COUNT = 42 / 3;
export const CATEGORIES = Array.from({ length: CATEGORY_COUNT }, (_, i) =>
  i.toString().padStart(2, "0")
);
