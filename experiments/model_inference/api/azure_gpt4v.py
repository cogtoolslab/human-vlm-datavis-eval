import requests 
import json 
from dotenv import load_dotenv
import os

load_dotenv()

api_base = os.getenv('AZURE_API_BASE')
deployment_name = "vis-benchmarker-2"
API_KEY = os.getenv('AZURE_API_KEY')

base_url = f"{api_base}/openai/deployments/{deployment_name}" 
headers = {   
    "Content-Type": "application/json",   
    "api-key": API_KEY 
} 
endpoint = f"{base_url}/chat/completions?api-version=2023-12-01-preview" 

def gpt4v(image_url, prompt, temperature, top_p, incontext_examples=None, num_trials=10):
	data = { 
	    "messages": [ 
		{ "role": "system", "content": "You are a helpful assistant." }, 
		{ "role": "user", "content": [  
		    { 
			"type": "text",
			"text": prompt
		    }
		] } 
	    ], 
	    "max_tokens": 2000,
		"temperature": temperature,
		"top_p": top_p,
	}
	if incontext_examples:
		for additional_example in incontext_examples:
			data["messages"][1]["content"].append(
				{ 
				"type": "image_url",
				"image_url": {
					"url": additional_example["image_url"]
				}
				}
			)

	# add main image
	data["messages"][1]["content"].append(
		{ 
		"type": "image_url",
		"image_url": {
			"url": image_url
		}
		}
	)

	print(endpoint, data)

	response = requests.post(endpoint, headers=headers, json=data)

	if response.status_code == 200:
		response = response.json()
		# print(response)
		model_res = response['choices'][0]["message"]["content"]
		return model_res
	if response.status_code == 429:
		print("Rate limit exceeded")

	return None

# if __name__ == "__main__":
	# res = gpt4v(
	# 	"https://data-visualization-benchmark.s3.us-west-2.amazonaws.com/davinci/images/movies/movies_x1genre_facetdecade_reorderalph_colorgrey20.png",
	# 	# "https://data-visualization-benchmark.s3.us-west-2.amazonaws.com/davinci/images/storms/storms_x1name_facetstatus_reorderalph_colorgrey20.png", 
	# 	"Please act as if you are a visualization expert. Can you respond to this question delimited by ///.\n///\nYou will be presented with a series of data visualizations, each accompanied by a question. Your goal is to answer each question as accurately and as quickly as you are able. It is common for people to not be fully sure when answering these questions, but please do your best on each question, even if you have to make a guess. How much higher are ratings of movies from the decade with the highest ratings compared to the decade with the lowest ratings?\n///\nAnswer:"
	# )
	# print(res)