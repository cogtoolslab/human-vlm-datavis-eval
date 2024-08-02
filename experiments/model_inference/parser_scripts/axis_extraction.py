import pandas as pd
import os
from dotenv import load_dotenv
import requests
import json
import time
import re
import ast



load_dotenv()

__label_extraction_prompt = """Please tell me if the x-axis contains number values. Please tell me if the y-axis has number values. Please list the values of the x-axis.  Please list the values of the y-axis. Please format your response to be the same as the examples shown below:
Example 1: {"x-axis-has-number": true, "y-axis-has-number": true, "x-axis": [1,2,3], "y-axis": [1,2,3], }
Example 2: {"x-axis-has-number": false, "y-axis-has-number": false, "x-axis": ['a', 'b', 'c'], "y-axis": ['a', 'b', 'c'], }
Example 3: {"x-axis-has-number": true, "y-axis-has-number": false, "x-axis": [1,2,3], "y-axis": ['a', 'b', 'c'], }
Example 4: {"x-axis-has-number": false, "y-axis-has-number": true, "x-axis": ['a', 'b', 'c'], "y-axis": [1,2,3]}
Answer:"""

api_base = os.getenv('AZURE_API_BASE')
deployment_name = "vis-benchmarker-2"
API_KEY = os.getenv('AZURE_API_KEY')

base_url = f"{api_base}/openai/deployments/{deployment_name}" 
headers = {   
    "Content-Type": "application/json",   
    "api-key": API_KEY 
} 
endpoint = f"{base_url}/chat/completions?api-version=2023-12-01-preview" 

def clean_json(raw_json_string):
    json_string = raw_json_string
    json_string = json_string.replace("\'", "\"")
    json_string = json_string.replace("'", "\"")

    # Regular expression to remove trailing commas before a closing bracket or brace
    json_string = re.sub(r',\s*([}\]])', r'\1', json_string)
    json_string = json_string.replace("true", "True")
    json_string = json_string.replace("false", "False")

    # fixed_dict = ast.literal_eval(json_string)

    return json_string

def extract_axis_labels(test_type):
    test_questions = pd.read_csv(f"https://data-visualization-benchmark.s3.us-west-2.amazonaws.com/{test_type}/questions.csv").copy()
    test_questions = test_questions[['image_link']].drop_duplicates()

    updates = []
    for i, row in test_questions.iterrows():
        response = call_gpt4v(row["image_link"])
        response = clean_json(response)
        # print("RESPONSE ", response )
        
        try:
            # response = json.loads(response)
            response = ast.literal_eval(response)
            # print("RESPONSE JSON", response)
            axis_list = []
            if response["y-axis-has-number"]:
                print("Y axis has numbers. List:", response["y-axis"])
                axis_list = response["y-axis"]
            elif response["x-axis-has-number"]:
                print("X axis has numbers. List:", response["x-axis"])
                axis_list = response["x-axis"]
            
                # print(f"Error parsing response: index: {i}, response: {response}")

            # print(row["image_link"])
            update = {
                'image': row["image_link"],
                'axis_list': axis_list,
                'numerical_axis_min': min(axis_list) if axis_list else None,
                'numerical_axis_max': max(axis_list) if axis_list else None,
                'y_axis_list': response["y-axis"],
                'x_axis_list': response["x-axis"],
                'y_axis_has_number': response["y-axis-has-number"],
                'x_axis_has_number': response["x-axis-has-number"],
            }
            updates.append(update)

        except:
            updates.append({
                'image': row["image_link"],
                'axis_list': None,
                'numerical_axis_min': None,
                'numerical_axis_max': None,
                'y_axis_list': None,
                'x_axis_list': None,
                'y_axis_has_number': None,
                'x_axis_has_number': None,
            })
            print("Error parsing response.")

        time.sleep(7)

    try:
        pd.DataFrame(updates).to_csv(f"{test_type}_axis_labels.csv")
    except:
        print("Error saving file")
    
    try: 
        updates_df = pd.DataFrame(updates, index=test_questions.index)
        test_questions.update(updates_df)
        test_questions.to_csv(f"{test_type}_questions.csv", index=False)
    except:
        print("Error saving file")

   

def call_gpt4v(image_url):
    deployment_name = "vis-benchmarker-2"
    base_url = f"{api_base}/openai/deployments/{deployment_name}" 
    endpoint = f"{base_url}/chat/completions?api-version=2023-12-01-preview" 

    data = { 
	    "messages": [ 
		{ "role": "system", "content": "You are a helpful assistant." }, 
		{ "role": "user", "content": [  
		    { 
			"type": "text",
			"text": __label_extraction_prompt
		    }
		] } 
	    ], 
	    "max_tokens": 2000,
		# "temperature": 0.6,
	}
    data["messages"][1]["content"].append(
        { 
        "type": "image_url",
        "image_url": {
            "url": image_url
        }
        }
    )

    response = requests.post(endpoint, headers=headers, json=data)

    if response.status_code == 200:
        response = response.json()
        model_res = response['choices'][0]["message"]["content"]
        return model_res
    if response.status_code == 429:
        print("Rate limit exceeded")

    return None


if __name__ == "__main__":
    for test_type in ['chartqa-test-continuous']:
        extract_axis_labels(test_type)