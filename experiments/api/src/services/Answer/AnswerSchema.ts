import { Schema, Document, model } from "mongoose";

export interface IAnswer extends Document {
  parsed_params: Object;
  survey: Object;
  order1: Object;
  order2: Object;
  order3: Object;
  order0: Object;
  order4: Object;
  order5: Object;
  order6: Object;
  order7: Object;
  order8: Object;
  order9: Object;
  order10: Object;
  order11: Object;
  order12: Object;
  order13: Object;
  order14: Object;
  order15: Object;
  order16: Object;
  order17: Object;
  order18: Object;
  order19: Object;
  order20: Object;
  order21: Object;
  order22: Object;
  order23: Object;
  order24: Object;
  order25: Object;
  study_end_timestamp: Number,
  study_start_timestamp: Number,
}

export const AnswerSchema = new Schema({
  parsed_params: { type: Object },
  survey: { type: Object },
  order1: { type: Object },
  order2: { type: Object },
  order3: { type: Object },
  order0: { type: Object },
  order4: { type: Object },
  order5: { type: Object },
  order6: { type: Object },
  order7: { type: Object },
  order8: { type: Object },
  order9: { type: Object },
  order10: { type: Object },
  order11: { type: Object },
  order12: { type: Object },
  order13: { type: Object },
  order14: { type: Object },
  order15: { type: Object },
  order16: { type: Object },
  order17: { type: Object },
  order18: { type: Object },
  order19: { type: Object },
  order20: { type: Object },
  order21: { type: Object },
  order22: { type: Object },
  order23: { type: Object },
  order24: { type: Object },
  order25: { type: Object },
  study_end_timestamp: {type: Number},
  study_start_timestamp: {type: Number},
  // question: { type: String },
  // answer: { type: Number },
  // correctAnswer: { type: String },
  // agentResponse: {type: String},
  // dataVisualizationUrl: { type: String },
  // serverAudioRecordingPath: {type: String},
  // taskCategory: { type: String },
  // dataVisualizationType: { type: String },
  // timestamp: { type: Number },
  // datasetTitle: {type: String},
  // imageDescription: {type: String},
  // units: {type: String},
  // trial: {type: Number },
});

export const Answer = model<IAnswer>("Answer", AnswerSchema);
