import { Router } from "express";
import { AnswerAPI } from "./Answer/AnswerAPI";

export class ComponentAPI {
  private router: Router;

  constructor(router: Router) {
    this.router = router;
  }
  public getRouter() {
    this.routeAnswer();
    return this.router;
  }

  private routeAnswer() {
    const answerAPI = new AnswerAPI(this.router);
    const answerRouter = answerAPI.returnRouter();
    this.router.use("/v1", answerRouter);
  }
}
