import { useState } from "react";
import { useNavigate } from "react-router-dom";

const protocolUnderstanding = [
  "To understand why many data visualizations are often misleading.",
  "To understand how people read graphs.",
  "To write incorrect answers.",
  "To understand how people think about data.",
];

const Instructions: React.FC<{
  nextUrl: string;
}> = ({ nextUrl }) => {
  const [consent, setConsent] = useState<string>("false");
  const [Q1, setQ1] = useState<string>("false");
  const [Q2, setQ2] = useState<string>("false");
  const [Q3, setQ3] = useState<string>("false");
  const [Q4, setQ4] = useState<string>("false");
  const navigate = useNavigate();

  const onClickContinue: React.MouseEventHandler<HTMLButtonElement> = () => {
    if (
      !(
        Q1 === "false" &&
        Q2 === "true" &&
        Q3 === "false" &&
        Q4 === "false" &&
        consent === "true"
      )
    ) {
      alert(
        "Please read the instructions carefully and check the appropriate boxes."
      );
      return;
    }
    navigate(nextUrl);
  };

  return (
    <div className="px-20 py-10 flex flex-col space-y-4">
      <h1 className="text-3xl font-bold">What does this graph say?</h1>

      <p>
        Welcome! We're researchers at Stanford University interested in how
        people read graphs.
      </p>
      <p>
        Over the next 20 minutes or so, we'll show you a few different graphs
        and ask you some questions about them.
      </p>
      <h2 className="text-2xl font-bold">Consent</h2>
      <p>
        By following the instructions, you are participating in a study being
        performed by researchers at Stanford University's Psychology Department.
        If you have questions about this research, please contact us at{" "}
        <a
          className="text-blue-700 underline"
          href="mailto:cogtoolslab.requester@gmail.com"
        >
          cogtoolslab.requester@gmail.com
        </a>
        . We will do our best to communicate with you in a timely, professional,
        and courteous manner. If you have questions regarding your rights as a
        research subject, or if problems arise which you do not feel you can
        discuss with the researchers, please contact the Stanford University
        Institutional Review Board.
      </p>

      <ul className="list-disc ml-10">
        <li>You must be at least 18 years old to participate.</li>
        <li>Your participation in this research is voluntary.</li>
        <li>
          You may decline to answer any or all of the following questions.
        </li>
        <li>
          You may decline further participation, at any time, without adverse
          consequences.
        </li>
        <li>
          Your anonymity is assured; the researchers who have requested your
          participation will not receive any personally identifying information.
        </li>
      </ul>

      <p>
        <input
          type="checkbox"
          className="ml-6 w-4 h-4 cursor-pointer"
          value={consent}
          onChange={() => setConsent((c) => (c === "false" ? "true" : "false"))}
        />
        <span> I have read and agree with the above.</span>
      </p>

      <h2 className="text-2xl font-bold">Before you start, please answer the following question: </h2>
      <p>What is the purpose of this study?</p>
      <div className="ml-6">
        <p>
          <input
            type="checkbox"
            className="w-4 h-4 cursor-pointer"
            value={Q1}
            onChange={() => setQ1((q) => (q === "false" ? "true" : "false"))}
          />
          <span> {protocolUnderstanding[0]}</span>
        </p>
        <p>
          <input
            type="checkbox"
            className="w-4 h-4 cursor-pointer"
            value={Q2}
            onChange={() => setQ2((q) => (q === "false" ? "true" : "false"))}
          />
          <span> {protocolUnderstanding[1]}</span>
        </p>
        <p>
          <input
            type="checkbox"
            className="w-4 h-4 cursor-pointer"
            value={Q3}
            onChange={() => setQ3((q) => (q === "false" ? "true" : "false"))}
          />
          <span> {protocolUnderstanding[2]}</span>
        </p>
        <p>
          <input
            type="checkbox"
            className="w-4 h-4 cursor-pointer"
            value={Q4}
            onChange={() => setQ4((q) => (q === "false" ? "true" : "false"))}
          />
          <span> {protocolUnderstanding[3]}</span>
        </p>
      </div>

      <button
        onClick={onClickContinue}
        className="bg-gray-200 hover:bg-gray-300 px-4 py-2 my-4 rounded-sm"
      >
        Continue
      </button>
    </div>
  );
};

export default Instructions;
