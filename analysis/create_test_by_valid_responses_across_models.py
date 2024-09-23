import pandas as pd
import numpy as np
import altair as alt
from evaluation_metrics import EvaluationMetrics
import os

AWS_PREFIX = "https://data-visualization-benchmark.s3.us-west-2.amazonaws.com"

def process_extracted_responses(model_configs, prompt_type='indist_instructions_question'):
    test_types = [
        "ggr",
        "vlat",
        "holf",
        'calvi-trick',
        'calvi-standard',
        'holf2',
        'chartqa-test-continuous',
        # 'chartqa-test-continuous-human',
    ]

    model_responses = []
    for model_config in model_configs:
        model = model_config['model']
        temperature_dir = model_config['temp']
        top_p_dir = model_config['top_p']
        _dir = f'responses/{prompt_type}/{top_p_dir}/{temperature_dir}/{model.replace("/", "-")}'
        for test_type in test_types:
            file_name = f'https://data-visualization-benchmark.s3.us-west-2.amazonaws.com/{test_type}/{_dir}/extracted_responses.csv'
            response_file = pd.read_csv(file_name)

            if test_type == 'chartqa-test-continuous-human':
                response_file['testType'] = 'chartqa-test-continuous-human'

            model_responses.append(response_file)

    processed_dfs = pd.concat(model_responses)

    questions = []
    for test_type in test_types:
        file_name = f"https://data-visualization-benchmark.s3.us-west-2.amazonaws.com/{test_type}/questions.csv"
        questions.append(pd.read_csv(file_name))

    questions = pd.concat(questions)

    metric = EvaluationMetrics()

    def find_non_floats(r):
        if r['res_len'] <= 2:
            res = str(r['agent_response'])
            res = res.replace(" ", "")
            res = res.replace(",", "")
            extracted_response = metric.extract_numerical_responses(res)
            if extracted_response:
                return extracted_response[0]
            else:
                return np.nan
        else:
            return np.nan
    
    def find_non_mc(r):
        item = questions[
            (questions["question"] == r["question"]) &
            (questions["image_file"] == r["image_file"])                
        ].iloc[0]
        response = str(r["agent_response"]).lower()
        mc1, mc2, mc3, mc4 = item["mc1"], item["mc2"], item["mc3"], item["mc4"]

        for mc in [mc4,mc3,mc2,mc1]:
            if mc != np.nan and response.find(str(mc).lower().strip()) != -1:
                return mc
    
        return np.nan

        
    def extract_valid_response(r):
        test_type = r['testType']
        if test_type in ['holf', 'holf2', 'chartqa-test-continuous', 'chartqa-test-continuous-human']:
            return find_non_floats(r)
        elif test_type in ["vlat", 'calvi-trick', 'calvi-standard']:
            return find_non_mc(r)
        elif test_type in ['ggr']:
            item = questions[
                (questions["question"] == r["question"]) &
                (questions["image_file"] == r["image_file"])                
            ].iloc[0]

            if item['text_input_type'] == 'open_ended_question':
                return find_non_floats(r)
            elif item['text_input_type'] == 'multiple_choice_question':
                return find_non_mc(r)
            else:
                return np.nan
    
    processed_dfs = processed_dfs.merge(questions[['image_file','question']])
    processed_dfs['res_len'] = processed_dfs['agent_response'].apply(
        lambda r : len(str(r).split())
    )
    processed_dfs['extracted_response'] = processed_dfs['agent_response'].copy()
    processed_dfs['agent_response'] = processed_dfs.apply(extract_valid_response, axis=1)

    model_responses = []
    for model_config in model_configs:
        model = model_config['model']
        temperature_dir = model_config['temp']
        top_p_dir = model_config['top_p']
        _dir = f'responses/{prompt_type}/{top_p_dir}/{temperature_dir}/{model.replace("/", "-")}'
        for test_type in test_types:
            mt_df = processed_dfs[
                (processed_dfs['agentType'] == model) &
                (processed_dfs['testType'] == test_type)
            ]
            # if not os.path.exists(_dir):
            #     os.makedirs(_dir)
            # file_name = f'{_dir}/{test_type}_processed_extracted_responses.csv'
            # mt_df.to_csv(file_name,index=False)
            # upload_file_to_s3(file_name, test_type, f'{_dir}/processed_extracted_responses.csv')

    charts = []
    for i, model_config in enumerate(model_configs):
        model = model_config['model']
        main_color = model_config['color']

        agent_responses =  processed_dfs[processed_dfs["agentType"] == model].copy()[["testType", "agent_response"]].reset_index()
        agent_responses['testType'] = agent_responses['testType'].replace({
            "calvi-trick": "calvi",
            "calvi-standard": "calvi"
        })

        agent_responses["is_valid"] = agent_responses["agent_response"].notna().astype(int)
        agent_responses = agent_responses.groupby(["testType"])["is_valid"].aggregate(lambda g : sum(g) / len(g))

        agent_responses = agent_responses.reset_index()
        
        test_order = [
            "ggr",
            "vlat",
            'calvi',
            "holf",
            'holf2',
            'chartqa-test-continuous',
            # 'chartqa-test-continuous-human',
        ]

        y_axis = None if i != 0 else alt.Axis()

        _dir = f"./prop_valid_responses/{model.replace("/", "_")}"
        if not os.path.exists(_dir):
            os.makedirs(_dir)
        agent_responses.to_csv(f"{_dir}/{test_type}.csv")

        chart = alt.Chart(agent_responses).mark_bar(color=main_color).encode(
            x=alt.X('testType', title=None, scale=alt.Scale(domain=test_order)),
            y=alt.Y('is_valid', title=None, axis=y_axis, scale=alt.Scale(domain=[0, 1]) ),
            color=alt.Color(legend=None),
        ).properties(width=80, height=100, title=alt.Title(
            "",
            anchor="middle"
        ))
        charts.append(chart)

    _dir = f"./prop_valid_responses"
    if not os.path.exists(_dir):
        os.makedirs(_dir)
    file_name = f'{_dir}/processed_responses_valid1.pdf'
    alt.hconcat(*charts, spacing=3).save(file_name)
    print(file_name)



if __name__ == "__main__":
    model_configs = [
        {'model': "Salesforce/blip2-flan-t5-xl", 'top_p': 'p06', 'temp': 't1', 'color': '#5ba3cf', 'accent_color': '#5ba3cf'},
        {'model': "Salesforce/blip2-flan-t5-xxl", 'top_p': 'p1', 'temp': 't10', 'color': '#4c78a8', 'accent_color': '#4c78a8'},
        {'model': "llava-hf/llava-1.5-7b-hf", 'top_p': 'p04', 'temp': 't1', 'color': '#f9b574', 'accent_color': '#f9b574'},
        {'model': "llava-hf/llava-1.5-13b-hf", 'top_p': 'p1', 'temp': 't04', 'color': '#f58518', 'accent_color': '#f58518'},
        {'model': "liuhaotian/llava-v1.6-34b", 'top_p': 'p1', 'temp': 't04', 'color': '#F1C232', 'accent_color': '#BF9000'},
        {'model': "google/pix2struct-chartqa-base", 'top_p': 'p08', 'temp': 't1', 'color': '#b9a7d0', 'accent_color': '#b9a7d0'},
        {'model': "google/matcha-chartqa", 'top_p': 'p04', 'temp': 't1', 'color': '#8b6db2', 'accent_color': '#8b6db2'},
        {'model': "GPT-4V", 'top_p': 'p1', 'temp': 't02', 'color': '#b85536', 'accent_color': '#b85536'},
    ]

    process_extracted_responses(model_configs=model_configs)
    