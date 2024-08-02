import { Router } from "express";
import { AnswerController } from "./AnswerController";
import multer from "multer";
import path from "path";
import fs from "fs";
import dotenv from "dotenv";

dotenv.config(); 

export class AnswerAPI {
  private router: Router;
  private controllerContext: AnswerController;

  constructor(router: Router) {
    this.router = router;
    this.controllerContext = new AnswerController();
  }

  public returnRouter() {
    this.postAnswer();
    this.listAnswer();
    this.postAnswerAudio();
    this.getCondition();
    return this.router;
  }

  private getCondition() {
    this.router.get("/condition", this.controllerContext.readConditions());
  }

  private listAnswer() {
    this.router.get("/answers", this.controllerContext.list());
  }

  private postAnswer() {
    this.router.post("/answer", this.controllerContext.create());
  }

  // private getAudio() {
  //   // list of completed session+prolifc id
  //   this.router.get("/answers/audio", this.controllerContext.list());
  // }

  private postAnswerAudio() {
    const uploadsDir =
      process.env.UPLOADS_DIR || path.join(__dirname, "../..", "audio");

    // Configure multer for audio file storage
    const storage = multer.diskStorage({
      destination: function (req, file, cb) {
        const participantDir = path.join(uploadsDir, req.body.sessionId);
        if (!fs.existsSync(participantDir)) {
          fs.mkdirSync(participantDir, { recursive: true });
        }
        // Set the destination folder for audio files
        cb(null, participantDir);
      },
      filename: function (req, file, cb) {
        // Use the original filename with current date to avoid name conflicts
        cb(
          null,
          file.originalname
        );
      },
    });

    const upload = multer({ storage: storage });

    this.router.post(
      "/answer/audio",
      upload.single("audio"),
      this.controllerContext.createAudio()
    );
  }

  private saveAnswer() {
    // this.router.put("/answer", this.controllerContext.update())
  }
}
