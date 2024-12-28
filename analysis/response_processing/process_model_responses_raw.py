import pandas as pd
import re
import numpy as np
from aws_upload import upload_file_to_s3
import os




class ModelResponse:

    def __init__(self) -> None:
        pass

    def process_model_responses(self, all_prompts_raw, 
                                agent_type='', 
                                test_type='', 
                                prompt_type='', 
                                temperature=1.0, 
                                top_p=1.0,
                                response_count_per_question=10) -> pd.DataFrame:
        print(f"Processing agent: {agent_type} prompt: {prompt_type}, test: {test_type}, top_p: {top_p}, temperature: {temperature}")
        questions = pd.read_csv(f"https://data-visualization-benchmark.s3.us-west-2.amazonaws.com/{test_type}/questions.csv")

        all_prompts = all_prompts_raw.copy()
    
        # replace python rounding 
        all_prompts = all_prompts.rename({
            'agentResponse': 'agent_response',
            'isCorrect': 'is_correct',
            'participant': 'participant_id',
            'imageFile': 'image_file',
            'testType': 'test_type',
            'agentType': 'agent_type'
        }, axis=1).replace({0.6000000000000001: 0.6}) 
        
        # replace strings to floats
        temp_num = float_hyperparam(temperature)
        top_p_num = float_hyperparam(top_p)
        

        all_prompts = all_prompts[
            (all_prompts['test_type'] == test_type)  &
            (all_prompts['promptType'] == prompt_type) & 
            (all_prompts['agent_type'] == agent_type) &
            (all_prompts['temperature'] == temp_num) &
            (all_prompts['topP'] == top_p_num)
        ]

        if test_type.find('calvi') != -1:
            all_prompts = process_calvi(all_prompts)
        elif test_type == "ggr":
            all_prompts = process_ggr(all_prompts)
        elif test_type == "vlat":
            all_prompts = process_vlat(all_prompts)
        

        given_responses = all_prompts.copy()

        responses = []
        for _, test_question_row in questions.iterrows():
            response_row = given_responses[
                (given_responses['question'] == test_question_row['question']) &
                (given_responses['image_file'] == test_question_row['image_file'])          
            ].copy()
            
            if (len(response_row) < 1):
                continue

            response_row = response_row.sort_values(by="timestamp", ascending=False)
            response_row['correct_answer'] = test_question_row['correct_answer']

            if (agent_type.find("Human") != -1):
                if (test_type == "ggr" or test_type == "vlat"):
                    if (agent_type == "Human/Math-2-1"):
                        item_responses = response_row.iloc[0:454]
                    elif (agent_type == "Human/Math-3"):
                        item_responses = response_row.iloc[0:632]
                elif (test_type == "holf"):
                    if (agent_type == "Human/Math-2-1"):
                        item_responses = response_row.iloc[0:284]
                    elif (agent_type == "Human/Math-3"):
                        item_responses = response_row.iloc[0:164]
                elif (test_type == "holf2"):
                    item_responses = response_row.iloc[0:response_count_per_question]
                elif (test_type == "calvi-trick" or test_type == "calvi-standard"):
                    item_responses = response_row.iloc[0:]
                else:
                    item_responses = response_row.iloc[0:response_count_per_question]
            else:
                item_responses = response_row.iloc[0:response_count_per_question]
            responses.append(item_responses)
        
        if (len(responses) == 0):
            print(f"Agent {agent_type} did not respond to any questions for test {test_type}")
            return pd.DataFrame()
        else:
            responses_df = pd.concat(responses)
            if ((len(responses_df) / len(questions)) != response_count_per_question):
                print(f"Agent {agent_type} only responded to {len(responses_df) / len(questions)} questions for test {test_type}")

        return responses_df
    


def float_hyperparam(hyperparam):
    if hyperparam.find("1") != -1:
        return 1
    else:
        return float(hyperparam.replace("t", "").replace("p", "")) / 10
    

def process_vlat(original_df):
    data = original_df.replace({
        'Over the course of years between 2009 and 2014, when was the number of girls named ‘Amelia’ at the maximum?':
        "Over the course of years between 2009 and 2014, when was the number of girls named 'Amelia' at the maximum?",
        "The number of girls named ‘Isla’ was __________ from 2009 to 2012.":
        "The number of girls named 'Isla' was __________ from 2009 to 2012.",
        "Over the course of years between 2009 and 2014, the number of girls named ‘Isla’ was always more than ‘Olivia’.":
        "Over the course of years between 2009 and 2014, the number of girls named 'Isla' was always more than 'Olivia'.",
        "In the UK, the number of girls named ‘Amelia’ in 2014 was more than it was in 2013.":
        "In the UK, the number of girls named 'Amelia' in 2014 was more than it was in 2013.",
        "What was the number of girls named ‘Amelia’ in 2010 in the UK?": 
        "What was the number of girls named 'Amelia' in 2010 in the UK?",
        "About what was the ratio of the number of girls named Olivia to those named Isla in 2014 in the UK?":
        "About what was the ratio of the number of girls named 'Olivia' to those named 'Isla' in 2014 in the UK?",
        "Which city’s metro system does lie outside the relationship between the total system length and the number of stations most?":
        "Which city's metro system does lie outside the relationship between the total system length and the number of stations most?"
    })

    return data

def process_ggr(original_df):
    data = original_df.replace({
        "What is the difference between the percentage of patients who recovered after a surgery and the percetage of patients who recovered after radiation therapy?'":
        "What is the difference between the percentage of patients who recovered after a surgery and the percentage of patients who recovered after radiation therapy?",
        "Approximately what percentage of people had Adeolitis in the year 2000":
        "Approximately what percentage of people had Adeolitis in the year 2000?",
        "Of 100 patients with disease X, how many are women":
        "Of 100 patients with disease X, how many are women?"
    })
    return data


def process_holf(original_df):
    data = original_df.replace({
        "davinci": "holf"
    })
    return data

def process_calvi(original_df):
    data = original_df.copy()
    def seprate_trick_items(r):
        if r["test_type"].find("calvi") != -1:
            if r["image_file"].find("T") != -1:
                return "calvi-trick"
            else:
                return "calvi-standard"
        else:
            return r['test_type']

    data['test_type'] = data.apply(seprate_trick_items, axis=1)
    return data

def process_chartqa_test(original_df):
    data = original_df.copy()
    questions = pd.read_csv(f"https://data-visualization-benchmark.s3.us-west-2.amazonaws.com/chartqa-test-categorical/questions.csv")
    
    def sperate_continuous_categorical(r):
        if r['test_type'].find("chartqa") != -1:
            if r['image_file'] in questions['image_file'].values and r['question'] in questions['question'].values:
                r['test_type'] = "chartqa-test-catgeorical"
            else:
                r['test_type'] = "chartqa-test-continuous"

        return r
    
    return data.apply(sperate_continuous_categorical, axis=1)


def read_model_configs():
    """Reads models with color scheme and hyperparameter values
    """
    model_configs = [
        {'model': "Salesforce/blip2-flan-t5-xl", 'top_p': 'p06', 'temp': 't1', 'color': '#5ba3cf', 'accent_color': '#8cbedd'},
        {'model': "Salesforce/blip2-flan-t5-xxl", 'top_p': 'p1', 'temp': 't10', 'color': '#4c78a8', 'accent_color': '#81a0c2'},
        {'model': "llava-hf/llava-1.5-7b-hf", 'top_p': 'p04', 'temp': 't1', 'color': '#f9b574', 'accent_color': '#facb9d'},
        {'model': "llava-hf/llava-1.5-13b-hf", 'top_p': 'p1', 'temp': 't04', 'color': '#f58518', 'accent_color': '#f79d46'},
        {'model': "liuhaotian/llava-v1.6-34b", 'top_p': 'p1', 'temp': 't04', 'color': '#BF9000', 'accent_color': '#F1C232'},
        {'model': "google/pix2struct-chartqa-base", 'top_p': 'p08', 'temp': 't1', 'color': '#ba94c4', 'accent_color': '#c7b8d9'},
        {'model': "google/matcha-chartqa", 'top_p': 'p04', 'temp': 't1', 'color': '#8b6db2', 'accent_color': '#a28ac1'},
        {'model': "GPT-4V", 'top_p': 'p1', 'temp': 't02', 'color': '#c93739', 'accent_color': '#C6765E'},
        {'model': "Human", 'top_p': 'pna', 'temp': 'tna', 'color': '#2e693b', 'accent_color': '#73D287'},
    ]
    return model_configs

def read_all_model_configs(models):
    """Reads models with all temps and p values
    """
    model_configs = []
    for model in models:
        top_p = 'p1'
        for temp in ['t02', 't04', 't06', 't08', 't10']:
            model_configs.append({
                'model': model, 
                'top_p': top_p, 
                'temp': temp,
            })

        temp = 't1'
        for top_p in ['p02', 'p04', 'p06', 'p08']:
            model_configs.append({
                'model': model, 
                'top_p': top_p, 
                'temp': temp,
            })

    return model_configs

if __name__ == "__main__":
    # all_prompts = pd.read_json("/Users/arnav/Downloads/prompts-full.json", low_memory=False)
    all_prompts = pd.read_json("/Users/arnav/Downloads/prompts_dec27_24.json")

    all_prompts = all_prompts[
        all_prompts['agentType'].apply(lambda a : a in [
            'meta-llama/Llama-3.2-11B-Vision', 
            'meta-llama/Llama-3.2-11B-Vision-Instruct', 
            'Xkev/Llama-3.2V-11B-cot',
            'ahmed-masry/chartgemma'
        ]) 
    ]
    all_prompts.to_json("/Users/arnav/Downloads/prompts_dec27_24_newmodels.json")
    
    # pd.read_json("/Users/arnav/Downloads/prompts_dec27_24.json")
    # all_prompts = pd.read_json("/Users/arnav/Downloads/prompts.json")
    prompt_types = [
        "indist_instructions_question",
    ]
    response_count_per_question = 10
    test_types = [
        # "ggr",
         "vlat",
        #  "holf",
        # "calvi",
        #  'calvi-trick', 
        #  'calvi-standard', 
        #  'holf2',
        # # 'chartqa-test-continuous',
        # 'chartqa-test-continuous-human',
    ]
    # model_configs = read_model_configs()
    model_configs = read_all_model_configs(models=['ahmed-masry/chartgemma', 'meta-llama/Llama-3.2-11B-Vision', 'meta-llama/Llama-3.2-11B-Vision-Instruct', 'Xkev/Llama-3.2V-11B-cot', ])
    print(len(all_prompts))
    print(all_prompts['agentType'].unique())
    model_response = ModelResponse()
    for prompt_type in prompt_types:
        for test_type in test_types:
            for mc in model_configs:
                agent_type = mc['model']
                temperature = mc['temp']
                top_p = mc['top_p']
                if agent_type == "Human":
                    continue

                agent_response_df = model_response.process_model_responses(
                    all_prompts, 
                    agent_type=agent_type,
                    prompt_type=prompt_type,
                    test_type=test_type,
                    temperature=temperature,
                    top_p=top_p,
                    response_count_per_question=response_count_per_question
                )

                top_p_dir = str(top_p).replace(".", "")
                temperature_dir= str(temperature).replace(".", "")
                # aws_dir = f'responses/{prompt_type}/{test_type}/{top_p_dir}/{temperature_dir}/{agent_type.replace("/", "-")}'
                aws_dir = f'responses/{prompt_type}/{top_p_dir}/{temperature_dir}/{agent_type.replace("/", "-")}'
                _dir = f'./{test_type}/{aws_dir}'
                file_name = f'{_dir}/raw_model_responses.csv'
                if not os.path.exists(_dir):
                    os.makedirs(_dir)

                print(_dir)
                agent_response_df.to_csv(file_name)
                upload_file_to_s3(file_name, test_type, f'{aws_dir}/raw_model_responses.csv')
                print("uploaded file")
