import { analyze } from "../analyze";
import { getData } from "../data";
import { trainKnnClassifier } from "./train";
import { validate } from "./validate";

getData()
  .then(({ training, validation }) =>
    trainKnnClassifier(training).then(classifier =>
      validate(classifier, validation)
    )
  )
  .then(analyze)
  .then(console.table);
