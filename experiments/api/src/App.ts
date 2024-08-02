import express, { Request, Response, NextFunction, Router } from "express";
import { Mongoose } from "mongoose";
import cors from "cors";
import path from "path";
import dotenv from "dotenv";
import cookieParser from "cookie-parser";
import morgan from "morgan";
import logger from "./utils/Logger";
import HTTPCodes from "./utils/HTTPCodes";
import { ComponentAPI } from "./services";
import MongooseODM from "./db/MongooseODM";
import { S3Client } from "@aws-sdk/client-s3";

const app = express();
dotenv.config(); // read env variables from .env file

/* Middlewares */
// block cross origin access
app.use(cors());
// convert all incoming requests to json
app.use(express.json({ limit: "10mb" }));
app.use(express.urlencoded({ extended: true, limit: "10mb" }));
app.use(cookieParser());

// show which routes are called in console
// only during development
if (process.env.NODE_ENV !== "development") {
  app.use(morgan("dev"));
}


// console.log(process.env.UPLOADS_DIR)

// connect to db
const mongooseODM = new MongooseODM();
export const db: Mongoose = mongooseODM.connectMongoDB();

// AWS S3 client
export const s3 = new S3Client({
  region: "us-west-2",
  credentials: {
    accessKeyId: process.env.AWS_ACCESS_KEY_ID!,
    secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY!,
  },
});

const router: Router = express.Router();
const componentAPI = new ComponentAPI(router);
app.use("/api", componentAPI.getRouter());

// print all api errors
app.use((err: Error, req: Request, res: Response, next: NextFunction) => {
  logger.error(err.message, err);
  return res.status(HTTPCodes.BAD_REQUEST).json({
    error: err.message,
  });
});

// conect to front end from external directory
const frontendBuildLocation = "../../frontend/dist/";
const frontendStaticLocation = path.join(__dirname, frontendBuildLocation);
app.use(express.static(frontendStaticLocation));
app.get("*", (req: Request, res: Response) => {
  res.sendFile(path.resolve(frontendStaticLocation, "index.html"));
});
console.log(frontendStaticLocation);

export default app;
