import { analyze } from "../analyze";
import { getData } from "../data";
import { trainSequentialModel } from "./train";
import { validate } from "./validate";

getData()
  .then(({ training, validation }) =>
    trainSequentialModel(training).then(model => validate(model, validation))
  )
  .then(analyze)
  .then(console.table);
