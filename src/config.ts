export const TRAINING_SIZE = 0.95;
export const VALIDATION_SIZE = 0.05;
export const CATEGORY_COUNT = 42;
export const CATEGORIES = Array.from({ length: 2 }, (_, i) =>
  i.toString().padStart(2, "0")
);
