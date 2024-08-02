export interface IVisualizationTask {
  question: string;
  units: string;
  instructions: string;
  task_id: string;
  set_id: string;
  image_link: string;
}

import { useState, useEffect } from "react";

// Custom hook that fetches tasks and returns it
function useFetchTasks(url: string): {
  tasks: IVisualizationTask[] | null;
  loading: boolean;
  error: Error | null;
} {
  const [tasks, setTasks] = useState<IVisualizationTask[] | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    // Function to fetchtasks
    const fetchtasks = async () => {
      try {
        const response = await fetch(url);
        if (!response.ok) {
          throw new Error("Network response was not ok");
        }
        const jsontasks = await response.json();
        const tasks: IVisualizationTask[] = jsontasks.tasks;
        setTasks(tasks);
      } catch (err) {
        if (err instanceof Error) {
          setError(err);
        }
      } finally {
        setLoading(false);
      }
    };

    // Call the fetch function
    fetchtasks();
  }, [url]); // Dependency array, ensures useEffect runs only if url changes

  return { tasks, loading, error };
}

export default useFetchTasks;
