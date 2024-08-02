import HTTPCodes from "./HTTPCodes";

export type HTTPError = NotFoundError | InternalServerError | BadRequestError;

export class NotFoundError extends Error {
  public HTTPCode: HTTPCodes;
  public name: string;

  constructor(message: string) {
    super(message);
    this.name = "HTTPNotFoundError";
    this.HTTPCode = HTTPCodes.NOT_FOUND;
  }
}

export class InternalServerError extends Error {
  public HTTPCode: HTTPCodes;
  public name: string;

  constructor(message: string) {
    super(message);
    this.name = "HTTPInternalServerError";
    this.HTTPCode = HTTPCodes.INTERNAL_SERVER_ERROR;
  }
}

export class BadRequestError extends Error {
  public HTTPCode: HTTPCodes;
  public name: string;

  constructor(message: string) {
    super(message);
    this.name = "HTTPBadRequestError";
    this.HTTPCode = HTTPCodes.BAD_REQUEST;
  }
}
