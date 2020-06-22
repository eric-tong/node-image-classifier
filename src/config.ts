export const TRAINING_SIZE = 0.95;
export const VALIDATION_SIZE = 0.02;
export const CATEGORY_COUNT = 5;
export const CATEGORIES = Array.from({ length: CATEGORY_COUNT }, (_, i) =>
  i.toString().padStart(2, "0")
);
