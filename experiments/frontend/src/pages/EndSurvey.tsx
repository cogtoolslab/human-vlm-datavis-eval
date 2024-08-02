import { useEffect, useState } from "react";
import toast from "react-hot-toast";
import { useNavigate } from "react-router-dom";

const CLASSES = [
  // Algebra 1, Geometry, Algebra 2, Trigonometry, Precalculus, Calculus 1, Calculus 2, Calculus 3 (Multivariable Calculus), Differential Equations, Linear Algebra, Probability and Statistics, Number Theory, Real Analysis, Abstract Algebra, None
  "Algebra",
  "Geometry",
  "Trigonometry",
  "Precalculus",
  "Calculus",
  "Multivariable Calculus",
  "Differential Equations",
  "Linear Algebra",
  "Probability and Statistics",
  "Computer Programming",
  "Real Analysis",
  "Data Science",
  "None of the above",
];

interface IEndSurveyProps {
  code: string;
  addResponse: (object: Object) => void;
  uploadResponse: () => Promise<void>;
}

const EndSurvey: React.FC<IEndSurveyProps> = ({
  // code,
  addResponse,
  uploadResponse,
}) => {
  const [education, setEducation] = useState("");
  const [graphMakingFrequency, setGraphMakingFrequency] = useState("");
  const [classesTaken, setClassesTaken] = useState({});
  const [experience, setExperience] = useState("");
  const [submitted, setSubmitted] = useState(false);
  // const [responseUploaded, setResponseUploaded] = useState(false);
  const navigate = useNavigate();

  const submit = async () => {
    // Cehck if all fields are filled
    if (
      education === "" ||
      graphMakingFrequency === "" 
    ) {
      alert("Please fill out all fields.");
      return;
    }
    addResponse({
      survey: {
        education,
        graphMakingFrequency,
        experience,
        classesTaken,
      },
      study_end_timestamp: new Date().getTime() 
    });

    setSubmitted(true);
  };

  useEffect(() => {
    if (submitted) {
      const res = uploadResponse();

      toast
        .promise(res, {
          loading: "Uploading results...",
          success: "Results uploaded!",
          error: "Error uploading audio. Please try again.",
        })
        // .then(() => {
        //   setResponseUploaded(true);
        // });
      // Navigate to the next page
      navigate("/end");
    }
  }, [submitted]);

  return (
    <div className="px-20 py-10 flex flex-col space-y-4">
      <h1 className="font-bold text-3xl">Thank you for your participation!</h1>
      <p>
        To finalize, we&apos;d like to know a little bit about your experience.
        All the questions require a response!
      </p>
      <p>
        What is the highest academic degree you have attained?{" "}
        <span className="text-red-500">* (required)</span>
        <select
          className="bg-gray-200 px-2 py-2 ml-2 rounded-md"
          onChange={(e) => setEducation(e.target.value)}
        >
          <option value="none">None</option>
          <option value="shs">Some high-school</option>
          <option value="hsd">High school diploma</option>
          <option value="sc">Some college</option>
          <option value="ba">Bachelor's degree in Arts (Non-STEM)</option>
          <option value="bs">Bachelor's degree in STEM</option>
          <option value="ma">Master's degree in Arts (Non-STEM)</option>
          <option value="ms">Master's degree in STEM</option>
          <option value="gp">
            Professional Degree (D.D.S., M.D., J.D., etc.)
          </option>
          <option value="gd">Doctorate (Ph.D., Ed.D., etc.)</option>
        </select>
      </p>
      <span>
        Which of the following classes have you taken?{" "}
        <span className="text-red-500">* (required)</span>
        <ul>
          {CLASSES.map((c) => (
            <li key={c}>
              <input
                className="bg-gray-200 px-2 py-1 rounded-md w-4 h-4"
                type="checkbox"
                value={c}
                onChange={(e) =>
                  setClassesTaken({ ...classesTaken, [c]: e.target.checked })
                }
              />
              {"  " + c}
            </li>
          ))}
        </ul>
      </span>
      <p>
        How comfortable are you with reading graphs?{" "}
        <span className="text-red-500">* (required)</span>
        <select
          className="bg-gray-200 px-2 py-2 ml-2 rounded-md"
          onChange={(e) => setGraphMakingFrequency(e.target.value)}
        >
          <option value="">Select</option>
          <option value="1">Not comfortable</option>
          <option value="2">Somewhat comfortable</option>
          <option value="3">Comfortable</option>
          <option value="4">Very comfortable</option>
        </select>
      </p>

      <p>
        Any feedback on how we can improve our experiment? For example, was the
        task too hard or too easy? Did you find the instructions unclear? Did
        you have any technical difficulties?
        <br />
        <textarea
          className="bg-gray-200 px-2 py-1"
          rows={5}
          cols={80}
          value={experience}
          onChange={(e) => setExperience(e.target.value)}
        />
      </p>
      {!submitted && (
        <button
          className="bg-gray-200 hover:bg-gray-300 rounded-lg p-2 mt-12"
          onClick={submit}
        >
          Finish Study
        </button>
      )}
    </div>
  );
};
export default EndSurvey;
