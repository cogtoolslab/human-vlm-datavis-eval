import { Request, Response } from "express";
import { BadRequestError, InternalServerError } from "../../utils/Errors";
import {
  sendHttpCreated,
  sendHttpError,
  sendHttpOk,
} from "../../utils/Responses";
import { Answer, IAnswer } from "./AnswerSchema";
import { s3 } from "../../App";
import { GetObjectCommand, ListObjectsCommand } from "@aws-sdk/client-s3";
import dotenv from "dotenv";
import csv from "csv-parser";
import fs from "fs";

dotenv.config();

const BASE_IMAGE_LINK =
  "https://data-visualization-benchmark.s3.us-west-2.amazonaws.com";
const AWS_BUCKET = "data-visualization-benchmark";
let condition_iteration = 0;
let conditions_completed = []

export class AnswerController {
  private num_items = 25;
  private stimuli_dir: string;

  constructor() {
    
    const stimli_dir = process.env.STIMULI_DIR;

    if (!stimli_dir) {
      throw new InternalServerError("Stimuli directory not found");
    }
    this.stimuli_dir = stimli_dir
    // this.experiment_dir = `${stimli_dir}/experiment_jsons`;
  }

  readConditions() {
    return async (req: Request, res: Response) => {
      const conditions: any[] = [];
      // grab all rows inbetween start and end (e.g first 25 rows)
      let count = 0;
      const start = this.num_items * condition_iteration;
      const end = this.num_items * (condition_iteration + 1);
      fs.createReadStream(`${this.stimuli_dir}/experiments.csv`)
        .pipe(csv())
        .on("data", (data) => {
          if (count < end && count >= start) {
            conditions.push(data);
          }
          count++;
        })
        .on("end", () => {
          // console.log(this.conditions);
          // Handle the parsed data here
          sendHttpOk(res, { tasks: conditions });
          // condition_iteration += 1
          // if (condition_iteration === 5) {
          //   condition_iteration = 0
          // }
        });
      
      // node read csv file
    };
  }

  list() {
    return async (req: Request, res: Response) => {};
  }

  results() {
    return async (req: Request, res: Response) => {};
  }

  createAudio() {
    return async (req: Request, res: Response) => {
      if (!req.file) {
        return res.status(400).send("No file uploaded.");
      }
      return res.send(`File uploaded successfully: ${req.file.filename}`);
    };
  }

  create() {
    return async (req: Request, res: Response) => {
      const {
        study_end_timestamp,
        study_start_timestamp,
        parsed_params,
        survey,
        order0,
        order1,
        order2,
        order3,
        order4,
        order5,
        order6,
        order7,
        order8,
        order9,
        order10,
        order11,
        order12,
        order13,
        order14,
        order15,
        order16,
        order17,
        order18,
        order19,
        order20,
        order21,
        order22,
        order23,
        order24,
        order25,
      } = req.body;

      // const imageUrl = `${BASE_IMAGE_LINK}/${testType}/images/${imageFile}`;

      try {
        const newAnswer = await Answer.create({
          study_end_timestamp,
        study_start_timestamp,
        parsed_params,
        survey,
        order0,
        order1,
        order2,
        order3,
        order4,
        order5,
        order6,
        order7,
        order8,
        order9,
        order10,
        order11,
        order12,
        order13,
        order14,
        order15,
        order16,
        order17,
        order18,
        order19,
        order20,
        order21,
        order22,
        order23,
        order24,
        order25,
        });

        if (newAnswer) {
          sendHttpCreated(res, "Study response profile created");
        } else {
          throw new InternalServerError("Couldn't create response profile");
        }
      } catch (err: any) {
        console.log("ERROR", err);
        sendHttpError(res, err.message);
      }
    };
  }
}
