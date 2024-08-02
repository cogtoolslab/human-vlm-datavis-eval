import { useNavigate } from "react-router-dom";

interface ISectionAlertProps {
  text: string;
  nextUrl: string;
}

const SectionAlert: React.FC<ISectionAlertProps> = ({  nextUrl }) => {
  const navigate = useNavigate();

  const onClickContinue: React.MouseEventHandler<HTMLButtonElement> = async (
    e
  ) => {
    e.preventDefault();
    navigate(nextUrl);
  };

  return (
    <div
      style={{ fontFamily: "Optima, sans-serif" }}
      className="px-20 py-10 flex flex-col h-screen"
    >
      <div className="my-auto">
        <div className="mx-auto text-4xl">
          {/* {text} */}
          Let's get started with the real questions. <br /> 
          You won't be told the correct answer after writing your responses.
        </div>
        <button
          onClick={onClickContinue}
          className="bg-slate-300 hover:bg-slate-200 px-4 py-2 mt-8 rounded-md w-full"
        >
          I'm ready to continue
        </button>
      </div>
    </div>
  );
};

export default SectionAlert;
