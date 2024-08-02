import requests 
import json 
from dotenv import load_dotenv
import os
import pandas as pd
import time

load_dotenv()

api_base = os.getenv('AZURE_API_BASE')
deployment_name = "vlm-datavis-grader"
API_KEY = os.getenv('AZURE_API_KEY')

base_url = f"{api_base}/openai/deployments/{deployment_name}" 
headers = {   
    "Content-Type": "application/json",   
    "api-key": API_KEY 
} 
endpoint = f"{base_url}/chat/completions?api-version=2023-12-01-preview" 

def create_question_prompt(question, agentResponse, choices=""):
	task_description = "Please read the following example. Then extract the answer from the model response and type it at the end of the prompt."
	if len(choices) == 0:
		hint = f"Hint: Please answer the question requiring a floating-point number with two decimal places and provide the final value, e.g., 1.23, 1.34, 1.45, at the end." # Please answer n/a if the answer does not have a a numerical value."
	else:
		hint = "Hint: Please answer the question and provide the correct option."
	question = f"Question: {question}"
	choices = f"Choices: {choices}\n" if choices else ""
	model_responses = f"Model response: {agentResponse}"
	end = "Extracted answer:"
	return f"{task_description}\n{hint}\n{question}\n{model_responses}\n{end}"

def gpt4_find_answer(question, agentResponse, choices):
	prompt = create_question_prompt(question, agentResponse, choices)
	data = { 
	    "messages": [ 
		{ "role": "system", "content": "You are a helpful assistant." }, 
		{ "role": "user", "content": [  
		    { 
			"type": "text",
			"text": prompt
		    }
		] 
        } 
	    ], 
	    "max_tokens": 2000 
	}
	# print(endpoint, data)

	response = requests.post(endpoint, headers=headers, json=data)
	# print(response.status_code, response)

	if response.status_code == 200:
		response = response.json()
		# print(response)
		try:
			model_res = response['choices'][0]["message"]["content"]
			return model_res
		except:
			return "RAW: " + agentResponse
	if response.status_code == 429:
		print("Rate limit exceeded")
	if response.status_code == 404:
		print("Wrong endpoint")

	return None

if __name__ == "__main__":
	# dataset = "ggr"
	# dataset = "chartqa-val"
	dataset = "holf2"
	# df = pd.read_csv(f"/Users/arnav/Desktop/contextvis/vlm-datavis-benchmark/analysis/tacl_analysis/data/all_model_responses.csv")
	df = pd.read_csv(f"/Users/arnav/Desktop/contextvis/vlm-datavis-benchmark/analysis/tacl_analysis/data/llava34b_model_responses.csv")
	df = df[
		# (df['agentType'] == 'GPT-4V') &  
		 (df["testType"] == dataset)]
	
	answers_df = pd.DataFrame(columns=list(df.columns) + ["answer"])
	question_meta = pd.read_csv(f"https://data-visualization-benchmark.s3.us-west-2.amazonaws.com/{dataset}/questions.csv")
	#(f"/Users/arnav/Desktop/contextvis/vlm-datavis-benchmark/analysis/tacl_analysis/data/{dataset}_gpt4v_model_responses.csv")
	
	count = 0
	for i, response in df.iterrows():
		row = question_meta[
				response["question_image"] == (question_meta["question"] + " & " + question_meta["image_file"])
		].iloc[0]
		textInput = row["text_input_type"]

		if textInput ==  "multiple_choice_question":
			row = question_meta[
				response["question_image"] == (question_meta["question"] + " & " + question_meta["image_file"])
			].iloc[0]
			question = row["question"]
			mc1, mc2, mc3, mc4 = row["mc1"], row["mc2"], row["mc3"], row["mc4"]
			multipleChoice = []
			for mc in [mc4,mc3,mc2,mc1]:
				if mc != "" and mc != "Skip" and isinstance(mc, str):
					multipleChoice.append(mc)
			mcString = ", ".join(multipleChoice)
		else:
			mcString = ""

		question = response["question_image"].split(" & ")[0]
		answer = gpt4_find_answer(question, response["agentResponse"], mcString)
		response_row = response
		response_row["answer"] = answer
		if (answer == None):
			break
		# print(answer)
		answers_df.loc[len(answers_df)] = response_row

		time.sleep(3)
		count += 1
		print("COUNT", count, answer)
		# if (i % 720 == 0):
		# 	time.sleep(60)
		# if (i % 2 == 0):
		# 	break
		
	answers_df.to_csv(f"/Users/arnav/Desktop/contextvis/vlm-datavis-benchmark/analysis/tacl_analysis/data/{dataset}_llava34b_responses_all.csv")
		# agentResponses = df[df["question"] == question]["agentResponse"].values
		# for response in agentResponses:
		    # create_question_prompt(row["question"], row["agentResponse"], mcString)

# /Users/arnav/Desktop/contextvis/vlm-datavis-benchmark/analysis/processed_agent_responses/
    # prompt_types = [
    #     "instructions_question", 
    #     # "indist_instructions_question"
    # ]
    # datasets=[
    #     # "ggr",
    #     # "vlat", 
    #     # "davinci", 
    # ]
    # models = [
    #     "GPT-4V"
    #     "Salesforce/blip2-flan-t5-xxl", 
    #     "Salesforce/blip2-flan-t5-xl",
    #     "llava-hf/llava-1.5-7b-hf", 
    #     "Salesforce/blip2-opt-6.7b"
    #     "Salesforce/blip2-opt-2.7b"
    # ]

    # for model in models:
    #     for dataset in datasets:
    #         for prompt_type in prompt_types:
    #             print(f"Running {model} on {dataset} with {prompt_type} prompts")
    #             if dataset == "davinci":
    #                 continue
    #             create_question_prompt()
