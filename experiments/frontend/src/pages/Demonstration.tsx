import { useNavigate } from "react-router-dom";
import React, { useState } from "react";

const Demonstration: React.FC<{
  nextUrl: string;
}> = ({nextUrl }) => {
  const navigate = useNavigate();
  const videoRef = React.useRef<HTMLVideoElement>(null);
  const [progress, setProgress] = useState(0);
  const [videoWatched, setVideoWatched] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);

  const onClickContinue: React.MouseEventHandler<HTMLButtonElement> = async (
    e
  ) => {
    e.preventDefault();
    navigate(nextUrl);
  };

  const handleTimeUpdate = () => {
    const video = videoRef.current;
    if (!video) return;
    const value = (100 / video.duration) * video.currentTime;
    setProgress(value);
  };

  const onVideoEnded = () => {
    // Called when the video playback ends
    setVideoWatched(true);
    setIsPlaying(false);
  };

  const onPlayVideoClick = () => {
    const video = videoRef.current;
    if (!video) return;
    if (video.paused || video.ended) {
      video.play();
      setIsPlaying(true);
    } else {
      video.pause();
      setIsPlaying(false);
    }
  };

  return (
    <div
      className="px-20 py-10 flex flex-col space-y-4"
      // className="px-10 py-4 flex flex-col"
    >
      <h1 className="text-3xl font-bold">What does this graph say?</h1>
      <p>
        In this experiment, you will:
        <ol className="list-decimal	ml-8 mt-2">
          <li>Listen to a video demonstrating what it means to talk aloud, </li>
          <li>See an example of a graph problem, and</li>
          <li>Complete 3 graph problems and talk aloud as you solve them.</li>
        </ol>
      </p>
      <h3 className="text-2xl font-bold">Video demonstration of talk aloud</h3>
      <p>We want you to talk aloud as you solve the problems.</p>
      <p>
        This means, saying out loud everything you would say to yourself
        silently as you are solving the problems. We're going to be recording
        your audio.
      </p>
      <p>
        To demonstrate what it means to talk aloud, watch the following video.
        This video is an example of a person talking aloud as they are solving a
        problem. Note that the problem in the video is not related to the
        problems you will solve in this experiment!
      </p>
      <div className="flex flex-col">
        <video
          ref={videoRef}
          onTimeUpdate={handleTimeUpdate}
          className="mx-auto rounded-md"
          style={{ width: "800px" }}
          onEnded={onVideoEnded}
        >
          <source
            src="https://datavis-verbal-reasoning.s3.us-west-2.amazonaws.com/Talk_Aloud_Demo.mp4"
            type="video/mp4"
          />
        </video>
        <div
          className="bg-gray-300 h-4 mt-2 rounded-md overflow-hidden mx-auto"
          style={{
            width: "800px",
          }}
        >
          <div
            className="bg-green-400"
            style={{
              width: `${progress}%`,
              height: "100%",
            }}
          />
        </div>
        <button
          onClick={onPlayVideoClick}
          className={`mx-auto ${
            isPlaying
              ? "bg-yellow-400 hover:bg-yellow-300"
              : "bg-emerald-500 hover:bg-emerald-400"
          } py-2 w-72 mt-4 rounded-md cursor-pointer `}
        >
          {isPlaying ? "Pause" : "Play"}
        </button>
      </div>

      {/* Print the the videoWatched state */}
      {/* <p>Video watched: {videoWatched ? "Yes" : "No"}</p> */}

      {/* <p>Click the 'Continue' button below once you are ready to proceed.</p> */}
      {!videoWatched && (
        <p>
          The 'Continue' button will appear after you have watched the video.
        </p>
      )}
      {videoWatched && (
        <>
          <button
            onClick={onClickContinue}
            disabled={!videoWatched}
            className={`${"bg-gray-200 hover:bg-gray-300"} px-4 py-2 mt-4 rounded-sm cursor-pointer`}
          >
            Continue
          </button>
        </>
      )}
    </div>
  );
};

export default Demonstration;
