import { BrowserRouter as Router, Route, Routes } from "react-router-dom";
import Instructions from "./pages/Instructions";
import EndSurvey from "./pages/EndSurvey";
import EndPage from "./pages/EndPage";
import Task from "./pages/Task";
import Demonstration from "./pages/Demonstration";
import SectionAlert from "./pages/SectionAlert";
import PracticeTask from "./pages/PracticeTask";
import { Toaster } from "react-hot-toast";
import useURLParameters from "./hooks/useURLParameters";
import useFetchTasks from "./hooks/useFetchTasks";
import { useResponse } from "./hooks/useResponse";
// import { useResponse } from "./hooks/useResponse";

function App() {
    const { addResponse, uploadResponse } = useResponse();
  const params = useURLParameters(addResponse);
  const { tasks } = useFetchTasks("/api/condition");

  // Throw an error if the SESSION_ID or USER_ID are not provided
  if ( (params.SESSION_ID === "null" || params.PROLIFIC_ID === "null") && (!import.meta.env.DEV) ) {
    // debugger
    // Display an error message as page
    return (
      <div>
        <h1>Error: SESSION_ID or PROLIFIC_ID not provided</h1>
        <p>
          Please make sure you are using the correct link to access the study.
        </p>
      </div>
    );
  }


  return (
    <Router>
      <Toaster />
      <Routes>
        <Route path="/" element={<Instructions nextUrl="/task" />} />
        <Route
          path="/demonstration"
          element={
            <Demonstration
              nextUrl={"/section_alert"}
            />
          }
        />
        <Route
          path="/section_alert"
          element={
            <SectionAlert
              text="Next, you'll complete a practice round."
              nextUrl="/practice_task"
            />
          }
        />
        <Route
          path="/practice_task"
          element={
            <PracticeTask
              nextUrl="/section_alert_real"
              addResponse={addResponse}
            />
          }
        />
        <Route
          path="/section_alert_real"
          element={
            <SectionAlert
              text="Let's get started with the real questions. You will not be told the correct answer after writing your responses."
              nextUrl="/task"
            />
          }
        />
        <Route
          path="/task"
          element={
            <Task
              tasks={tasks || []}
              addResponse={addResponse}
            />
          }
        />
        <Route
          path="/survey"
          element={
            <EndSurvey
              code="CSUVIWY7"
              addResponse={addResponse}
              uploadResponse={uploadResponse}
            />
          }
        />
        <Route
          path="/end"
          element={<EndPage code="CSUVIWY7" />}
        />
      </Routes>
    </Router>
  );
}

export default App;
