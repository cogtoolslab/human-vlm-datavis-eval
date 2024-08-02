import time
from .data_utils import DataUtils
from .model_utils import ModelUtils
import pandas as pd

class Benchmark:
    def __init__(self, agentType, testType, promptType, numTrials=10, timeout=0, temperature=1, topP=1.0):
        self.agentType = agentType
        self.testType = testType
        self.promptType = promptType
        self.numTrials = numTrials
        self.timestamp = int(time.time() * 1000)
        self.timeout = timeout

        self.instructions = DataUtils.fetch_instructions(self.testType)
        self.process_func = ModelUtils(self.agentType, temperature, topP).process_image
        self.temperature = temperature
        self.topP = topP

    def has_gpt4v_returned_invalid(self, row):
        invalid_df = pd.read_csv("/Users/arnav/Desktop/contextvis/vlm-datavis-benchmark/analysis/gpt4v_invalid_prompt_jan19.csv")
        is_question = (row["question"] + " & " + row["image_file"])  not in invalid_df["question_image"].values
        # is_prompt_type = self.promptType not in invalid_df["promptType"].values
        return is_question #and is_image # and is_prompt_type

    def run(self):
        df = DataUtils.fetch_questions(self.testType)
        print(f"Currently on test {self.testType} for agent: {self.agentType} for top_p={self.topP}, t={self.temperature}.")

        if (self.timeout > 0):
            for i, _ in df.iterrows():
                # if self.has_gpt4v_returned_invalid(row):
                #    continue
                # if (i < 286):
                #     continue
                print(f"INDEX {i}")
                #idx = int(len(df) - 1 - i)
                row = df.iloc[i]
                self.benchmark_question(row)
        else:
            df.apply(self.benchmark_question, axis=1)

    def benchmark_question(self, row):
        prompt, incontext_examples = self.create_prompt_from_row(row)
        textInput = row["text_input_type"]
       
        for i in range(self.numTrials):
            generated_text = self.process_func(row["image_link"], prompt, incontext_examples=incontext_examples)
            print(f"Benchmark Trial {i}. Question index: {row.name}. Generated text: {generated_text}")

            data = {
                "question": row["question"],
                "prompt": prompt,
                "agentType": self.agentType,
                "testType": self.testType,
                "correctAnswer": row["correct_answer"],
                "agentResponse": generated_text,
                "imageFile": row["image_file"],
                "imageLink": row["image_link"],
                "taskCategory": "",
                "multipleChoice": [],
                "metadataLink": "",
                "timestamp": self.timestamp,
                "textInput": textInput,
                "promptType": self.promptType,
                "temperature": self.temperature,
                "topP": self.topP
            }

            DataUtils.post_data(data)

            if self.timeout > 0:
                time.sleep(self.timeout)

        return data


    def create_prompt_from_row(self, row):
        textInput = row["text_input_type"]
        question = row["question"]

        if textInput == "open_ended_question":
            # restriction = "Your answer must be numerical and rounded to 3 significant figures."
            # prompt = restriction + " " + question
            prompt = question
        
        elif textInput == "multiple_choice_question":
            mc1, mc2, mc3, mc4 = row["mc1"], row["mc2"], row["mc3"], row["mc4"]
            multipleChoice = []
            for mc in [mc4,mc3,mc2,mc1]:
                if mc != "" and mc != "Skip" and isinstance(mc, str):
                    multipleChoice.append(mc)
            mcString = ", ".join(multipleChoice)
            restriction = "Your answer must be one of the choices provided."
            prompt = restriction + " " + question + " " + "Choices: " + mcString + "."

        if self.agentType == "GPT-4V":
            promptPrefix = "Question: "
            promptSuffix = "\nAnswer:"
        elif self.agentType == "Salesforce/blip2-opt-2.7b":
            promptPrefix = "Question: "
            promptSuffix = " Answer:"
        elif self.agentType == "Salesforce/blip2-opt-6.7b":
            promptPrefix = "Question: "
            promptSuffix = " Answer:"
        elif self.agentType == "Salesforce/blip2-flan-t5-xxl":
            promptPrefix = "Question: "
            promptSuffix = "\nAnswer:"
        elif self.agentType == "Salesforce/blip2-flan-t5-xl":
            promptPrefix = "Question: "
            promptSuffix = "\nAnswer:"           
        elif (self.agentType == "llava-hf/llava-1.5-7b-hf" or self.agentType == "llava-hf/llava-1.5-13b-hf"):
            promptPrefix = "USER: <image>\n"
            promptSuffix = "\nASSISTANT:"
        elif (self.agentType == "liuhaotian/llava-v1.6-34b" or 
              self.agentType == "liuhaotian/llava-v1.6-vicuna-13b" or 
              self.agentType == "liuhaotian/llava-v1.6-mistral-7b"):
            promptPrefix = "USER: "
            promptSuffix = "\nASSISTANT:"
        elif (self.agentType == "google/pix2struct-chartqa-base" or 
               self.agentType == "google/matcha-chartqa"):
            promptPrefix = "Question: "
            promptSuffix = " Answer:"

        if self.promptType == "indist_instructions_question":
            prompt = promptPrefix + self.instructions + " " + prompt + promptSuffix
            if (self.agentType == "google/deplot"):
                prompt = "Generate underlying data table of the figure below:"

        elif self.promptType == "indist_instructions_cot_0shot_question":
            cot_0shot = " Let's think step by step."
            prompt = promptPrefix + self.instructions + " " + prompt + promptSuffix + cot_0shot

        elif self.promptType == "instructions_question":
            prompt = self.instructions + " " + prompt

            if (self.agentType == "llava-hf/llava-1.5-7b-hf" or self.agentType == "llava-hf/llava-1.5-13b-hf"):
                prompt = "<image>\n" + prompt

        elif self.promptType == "incontext_instructions_question":
            if self.testType.find("chartqa") != -1:
                incontext_examples = [
                    {
                        "image_url": 'https://data-visualization-benchmark.s3.us-west-2.amazonaws.com/chartqa-val/images/multi_col_133.png',
                        "question": 'How many people are forecast to be occasional viewers of eSports by 2024?',
                        "answer": '291.6'
                    },
                    {
                        "image_url": 'https://data-visualization-benchmark.s3.us-west-2.amazonaws.com/chartqa-val/images/two_col_100022.png',
                        "question": 'What was the second-most-shared news page on Facebook in January 2017?',
                        "answer": 'CNN'
                    },
                    {
                        "image_url": 'https://data-visualization-benchmark.s3.us-west-2.amazonaws.com/chartqa-val/images/two_col_6132.png',
                        "question": 'How many widowed people lived in Canada in 2000?',
                        "answer": '1.55'
                    },
                    {
                        "image_url": 'https://data-visualization-benchmark.s3.us-west-2.amazonaws.com/chartqa-val/images/two_col_1365.png',
                        "question": 'What percentage of people infected with the coronavirus were female?',
                        "answer": '51.1'
                    }
                ]

                if self.agentType.find("llava") != -1:
                    ex1 = "USER: <image>\n Q: " + incontext_examples['question'][0] + " A: ASSISTANT: " + incontext_examples[0]["answer"] + "." 
                    ex2 = "USER: <image>\n Q: " + incontext_examples['question'][1] + " A: ASSISTANT: " + incontext_examples[1]["answer"] + "." 
                    ex3 = "USER: <image>\n Q: " + incontext_examples['question'][2] + " A: ASSISTANT: " + incontext_examples[2]["answer"] + "." 
                    ex4 = "USER: <image>\n Q: " + incontext_examples['question'][3] + " A: ASSISTANT: " + incontext_examples[3]["answer"] + "." 
                    intstruction = "Following these examples, please answer the following question using either a word or a number."
                    prefix = "Q:"
                    suffix = "A: ASSISTANT:"
                    prompt = f"{ex1} {ex2} {ex3} {ex4} {intstruction} {prefix} {prompt} {suffix}"
                    return prompt, incontext_examples
                elif self.agentType == "GPT-4V":
                    incontext_examples = [
                        {
                            "image_url": 'https://data-visualization-benchmark.s3.us-west-2.amazonaws.com/chartqa-test/chartqa_incontext.png'
                        }
                    ]
                    system_message = "Suppose that you are an expert in data analysis and visualization." 
                    ex = "The first image is titled Examples. Following the examples shown in the first image, please answer the following question about the second image, using either a word or a number."
                    prefix = "Q:"
                    prompt = f"{system_message} {ex} {prefix} {prompt}"
                    return prompt, incontext_examples
        else:
            return None
            
        return prompt, None
