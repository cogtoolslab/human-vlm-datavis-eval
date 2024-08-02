import { useState, useEffect } from "react";

function useURLParameters(addResponse: (obj: Object) => void) {
  const [params, setParams] = useState({
    PROLIFIC_ID: "null",
    SESSION_ID: "null",
    STUDY_ID: "null",
  });

  useEffect(() => {
    const searchParams = new URLSearchParams(window.location.search);
    const PROLIFIC_ID = searchParams.get("PROLIFIC_PID"); // Note the change from 'PROLIFIC_ID' to 'PROLIFIC_PID'
    const SESSION_ID = searchParams.get("SESSION_ID");
    const STUDY_ID = searchParams.get("STUDY_ID");

    const parsed_params = {
      PROLIFIC_ID: PROLIFIC_ID || "null",
      SESSION_ID: SESSION_ID || "null",
      STUDY_ID: STUDY_ID || "null",
    };
    addResponse({ parsed_params, study_start_timestamp: new Date().getTime() });
    setParams(parsed_params);
  }, []);

  return params;
}

export default useURLParameters;
