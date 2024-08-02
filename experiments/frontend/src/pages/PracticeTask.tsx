import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import "./task.css";
import { IVisualizationTask } from "../hooks/useFetchTasks";

const visualizationTask: IVisualizationTask = {
  image_link:
    "https://datavis-verbal-reasoning.s3.us-west-2.amazonaws.com/images/penguin_pie_chart.svg",
  question:
    "How many times larger is the population of Adelie penguins compared to that of Chinstrap penguins?",
  units: "times",
  instructions:
    "This is a practice round for you to try out before starting the study. Because graphs can be complicated to look at, you'll be shown a description about the graph above the image. There will be a question just below the image.",
  task_id: "practice",
  set_id: "practice",
};

const PracticeTask: React.FC<{
  nextUrl: string;
  addResponse: (object: Object) => void;
}> = ({
  nextUrl,
  addResponse,
}) => {
  const navigate = useNavigate();
  const [timeElapsed, setTimeElapsed] = useState(0);
  const [timerActive, setTimerActive] = useState(false);
  const [answer, setAnswer] = useState<number | null>(null);
  const [timeState, setTimeState] = useState<{
    timestamp_start: number;
    local_time_start: string;
    time_limit: number;
  }>()

  useEffect(() => { 
    const timeState = {
      timestamp_start: new Date().getTime(),
      local_time_start: new Date().toLocaleTimeString('en-US'),
      time_limit: 20000,
    };
    setTimeState(timeState)
  }, [])

  useEffect(() => {
    let interval: any = 0;

    if (timerActive) {
      const startTime = Date.now() - timeElapsed;
      interval = setInterval(() => {
        const time = Date.now() - startTime;
        setTimeElapsed(time);
      }, 1000);
    }

    return () => clearInterval(interval);
  }, [timerActive, timeElapsed]);

  const startTimer = () => {
    setTimerActive(true);
  };

  const stopTimer = () => {
    setTimerActive(false);
  };

  useEffect(() => {
    console.log("Starting record");
    startTimer();
  }, []);

  const uploadTaskAudio = async () => {
    const task_id = visualizationTask?.task_id;

    if ( !task_id || !timeState) {
      return;
    }
    const order_id = `order0`;
    const timestamp_end = new Date().getTime();
    addResponse({
      [order_id]: {
        ...visualizationTask,
        order_id,
        answer,
        time_elapsed: timestamp_end - timeState.timestamp_start,
        ...timeState,
        local_time_end: new Date().toLocaleTimeString('en-US'),
        timestamp_end: new Date().getTime(),
      },
    });
    navigate(nextUrl);
  };

  const onSubmitClick: React.MouseEventHandler<
    HTMLButtonElement
  > = async () => {
    if (answer === null) {
      alert("Please input a number before submitting.");
      return;
    }
    stopTimer();
    uploadTaskAudio();
    window.scrollTo(0, 0);

    if (answer) {
      const correctAnswer = 2.22;
      const diff = Math.abs(correctAnswer - answer);
      const answerDiff = Math.round(diff * 100) / 100;
      alert(
        `The correct answer is ${correctAnswer} times. You were off by around ${answerDiff} from the answer. `
      );
    }
  };

  return (
    <div id="overall" className="px-20 py-10 flex flex-col">
      <h1 className="text-4xl font-bold">What does this graph say?</h1>
      <p className="mt-4">
        Take a look at the graph below and try to answer the question as quickly and as accurately as you can.
        <br /> 
      </p>
      <div
        style={{ width: "900px" }}
        className="flex flex-col border mx-auto p-6 bg-white rounded-lg shadow-lg mt-6"
      >
        <img
          className="mx-auto"
          style={{ width: "800px", height: "500px" }}
          src={visualizationTask.image_link}
          alt={visualizationTask.image_link}
        />
        <div className="font-bold">Question: </div>
        <div>{visualizationTask.question}</div>
        <div className="font-bold">Answer: </div>
        <div className="mb-4">
          Input only a number. Do <u>not</u> use any special symbols (eg. $, %).
          The submission button will be disabled if you input anything other
          than a number.
        </div>
        <div>
          Please try to be as accurate as possible! You may use decimals in your
          response.
        </div>
        <div>
          <input
            onChange={(e) => setAnswer(parseFloat(e.target.value))}
            className="border border-black px-2 rounded-sm"
            type="number"
          />{" "}
          {" " + visualizationTask.units}
        </div>

        {(
          <button
            // disabled={Math.floor(timeElapsed / 1000) < 20 || answer === null}
            onClick={onSubmitClick}
            className={`${
              Math.floor(timeElapsed / 1000) < 20
                ? "bg-gray-100 hover:bg-gray-300"
                : "bg-gray-300 hover:bg-gray-400 cursor-pointer"
            } px-4 py-2 mt-4 rounded-sm`}
          >
            Submit Answer
          </button>
        )}
      </div>
    </div>
  );
};

export default PracticeTask;
