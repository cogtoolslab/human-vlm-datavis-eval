import { Response } from "express";
import HTTPCodes from "./HTTPCodes";

export function sendHttpOk(res: Response, message: any) {
  res.status(HTTPCodes.OK).send(message);
}

export function sendHttpCreated(res: Response, message: any) {
  res.status(HTTPCodes.CREATED).send(message);
}

export function sendHttpDeleted(res: Response, message: any) {
  res.status(HTTPCodes.ACCEPTED).send(message);
}

export function sendHttpUpdated(res: Response) {
  res.status(HTTPCodes.UPDATED).send();
}

export function sendHttpError(res: Response, err: any) {
  err.errorCode
    ? res.status(err.errorCode).send(err.message)
    : res.status(HTTPCodes.INTERNAL_SERVER_ERROR).send(err.message);
}
