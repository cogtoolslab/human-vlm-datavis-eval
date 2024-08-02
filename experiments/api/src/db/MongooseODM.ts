import mongoose, { Mongoose } from "mongoose";

/* 
  MongooseODM object which give 
  handles connection to the database 
*/
export default class MongooseODM {
  constructor() {}

  /* connect mongoose to the appropriate MongoDB instance  */
  public connectMongoDB(): Mongoose {
    if (process.env.NODE_ENV === "test") {
      this.connectWithEnv(true).catch((error: NodeJS.ErrnoException) => {
        console.error.bind(error, "Error Connecting to MongoDB:");
      });
    } else {
      this.connectWithEnv(false).catch((error: NodeJS.ErrnoException) => {
        console.error.bind(error, "Error Connecting to MongoDB:");
      });
    }
    return mongoose as Mongoose;
  }

  /* Connect to mongoDB based on enviroment vars
  Mainly used to connect to external db */
  private async connectWithEnv(testingEnv: Boolean) {
    const mongooseOptions = {
      
    };
    const mongoDbUrl = testingEnv
      ? process.env.MONGODB_TEST_URL
      : process.env.MONGODB_URL;
    if (!mongoDbUrl) {
      throw new Error("MonogoDB URL not defined in .env file");
    }
    const mongooseODM = await this.connectMongoose(mongoDbUrl, mongooseOptions);

    return mongooseODM;
  }

  /* Helper to connect mongoDB to mongoose */
  private async connectMongoose(
    mongoDbUrl: string,
    mongooseOptions: any
  ) {
    const mongooseODM: Mongoose = await mongoose.connect(
      mongoDbUrl,
      mongooseOptions
    );
    return mongooseODM;
  }
}
