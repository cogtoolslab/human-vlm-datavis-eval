import requests 
import json 
from dotenv import load_dotenv
import os

load_dotenv()

api_base = os.getenv('AZURE_API_BASE')
deployment_name = "table-responder"
API_KEY = os.getenv('AZURE_API_KEY')

base_url = f"{api_base}/openai/deployments/{deployment_name}" 
headers = {   
    "Content-Type": "application/json",   
    "api-key": API_KEY 
} 
endpoint = f"{base_url}/chat/completions?api-version=2023-12-01-preview" 

def gpt4(prompt, temperature, top_p, incontext_examples=None, max_tokens=2000):
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
	    "max_tokens": max_tokens,
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