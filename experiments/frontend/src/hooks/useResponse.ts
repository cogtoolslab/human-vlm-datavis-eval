import { useState } from "react";

export const useResponse = () => {
  const [response, setResponse] = useState<Object>({});
  console.log(response);

  const addResponse = (object: Object) => {
    setResponse((prevState) => {
      return {
        ...prevState,
        ...object,
      };
    });
  };

  const uploadResponse = async () => {
    await fetch("/api/answer", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(response),
    });
  };

  return { addResponse, uploadResponse };
};
