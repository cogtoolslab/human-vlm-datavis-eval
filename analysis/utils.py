import pandas as pd
from typing import List, Dict, Literal

IData: List[Dict[str, str]] = [
    {
        'model': str,
        'top_p': Literal["p06", "p1", "p04", "p08", "pna"],
        'temp': Literal["t1", "t10", "t04", "t02", "tna"],
        'color': str,
        'accent_color': str
    }
]

def read_model_configs():
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

def get_all_test_types():
    return ['ggr', 'vlat', 'holf', 'calvi-trick', 'holf2', 'chartqa-test-continuous']

def create_all_agent_response_dataframe(
    test_name, 
    prompt_type="indist_instructions_question",
    model_configs=[],
    print_stats=False,
    response_file='processed_extracted_responses',
    dropna=True
):
    AWS_PREFIX = "https://data-visualization-benchmark.s3.us-west-2.amazonaws.com"
    questions = pd.read_csv(f'{AWS_PREFIX}/{test_name}/questions.csv')

    model_responses = []
    for model_config in model_configs:
        top_p_dir = model_config['top_p']
        temperature_dir = model_config['temp']
        model = model_config['model'].replace("/", "-")

        _dir = f'responses/{prompt_type}/{top_p_dir}/{temperature_dir}'
        if model == "Human":
            current_response_file = "processed_extracted_responses"
        else:
            current_response_file = response_file

        if response_file == "model_responses" and test_name == "chartqa-test-continuous-human":
            if model != "Human":
                test_type = "chartqa-test-continuous"
            else:
                test_type = "chartqa-test-continuous-human"
        else:
            test_type = test_name


        model_response_url = f'https://data-visualization-benchmark.s3.us-west-2.amazonaws.com/{test_type}/{_dir}/{model}/{current_response_file}.csv'
        model_response = pd.read_csv(model_response_url, low_memory=False)
        try:
            model_response['testType'] = model_response['test_type']
        except:
            pass
        
        # combine calvi-trick to be a part of calvi-standard
        if test_type == "calvi-trick":
            def get_calvi_standard():
                test_name_new = "calvi-standard"
                model_response_url = f'https://data-visualization-benchmark.s3.us-west-2.amazonaws.com/{test_name_new}/{_dir}/{model}/{current_response_file}.csv'
                model_response = pd.read_csv(model_response_url)
                try:
                    model_response['testType'] = model_response['testType'].replace("calvi-standard", "calvi-trick")
                except KeyError:
                    model_response['testType'] = model_response['test_type'].replace("calvi-standard", "calvi-trick")
                return model_response
            calvi_standard_df = get_calvi_standard()
            calvi_df = pd.concat([calvi_standard_df, model_response]).reset_index(drop=True)
            model_responses.append(calvi_df)
        else:
            model_responses.append(model_response)

        if test_name == "chartqa-test-continuous-human":
            model_response['testType'] = model_response['testType'].replace("chartqa-test-continuous", 
                                                                            "chartqa-test-continuous-human")
            
    model_responses = pd.concat(model_responses).reset_index(drop=True)
    
    
    if test_type != "calvi-trick":
        model_responses = model_responses[model_responses["testType"] == test_type]
    
    if 'image_file' not in model_responses.columns:
        model_responses = model_responses.rename({
            "imageFile": "image_file"
        }, axis=1)
    if 'agentType' not in model_responses.columns:
        model_responses['agentType'] = model_responses['agent_type']
        # model_responses = model_responses.rename({
        #         "agent_type": "agentType"
        #     }, axis=1)
    else:
        # DEBUG this is somethign doesnt work
        model_responses['agent_type'] = model_responses['agentType']
    
    model_responses = model_responses.drop(['correct_answer'], axis=1)
    model_responses = model_responses.merge(questions[['image_file','question','correct_answer']])
    if dropna:
        model_responses = model_responses.dropna(subset=["agent_response"])

    if print_stats:
        print(model_response_url)
        for model_config in model_configs:
            agent_type = model_config['model']
            agent_responses = model_responses[model_responses["agentType"] == agent_type]
            print(f"Agent {agent_type} only responded to {len(agent_responses) / len(questions)} questions for test {test_type}")

    cols = ['image_file', 'question', 'correct_answer', 'agent_response', 'testType', 'agentType']
    
    # Do this mainly for human participants -- e.g. we need to do this for ggr
    if 'is_correct' in model_responses.columns:
        cols.append('is_correct')
    return model_responses[cols]

def minmax_normalized_error(response_error, group_min, group_max):
        """Calculate the error rate between the response and the correct answer."""
        return (response_error - group_min) / (group_max - group_min)

def get_absolute_error(numerical_answer : float, correct_numerical_answer : float):
    return abs(numerical_answer - correct_numerical_answer) 

def find_min_max_scaled_error(model_responses_raw, test_name, units_of_measure):
    model_responses = model_responses_raw.copy()
    model_responses["agent_response"] = model_responses["agent_response"].astype(float)
    AWS_PREFIX = "https://data-visualization-benchmark.s3.us-west-2.amazonaws.com"
    questions = pd.read_csv(f'{AWS_PREFIX}/{test_name}/questions.csv')

    merged_response = questions[["question", "image_file", "min_label", "max_label"]]
    model_responses = model_responses.merge(merged_response, on=["question", "image_file"])
    model_responses["agent_response"] = model_responses.apply(
        lambda r: minmax_normalized_error(r['agent_response'], r['min_label'], r['max_label']), axis=1
    )
    model_responses["correct_answer"] = model_responses.apply(
        lambda r: minmax_normalized_error(r['correct_answer'], r['min_label'], r['max_label']), axis=1
    )
    model_responses[units_of_measure] = model_responses.apply(
        lambda r: get_absolute_error(r['agent_response'], r['correct_answer']), axis=1
    )
    model_responses = model_responses.dropna(subset=[units_of_measure])
    return model_responses

def find_correct_mc(model_responses_raw):
    model_responses = model_responses_raw.copy()

    process_answer = lambda a : str(a).lower().strip()
    def check_answer(r):
        if (r['testType'] in ['ggr', 'vlat']) and r['agentType'] == 'Human':
            return int(r['is_correct'])

        is_correct = process_answer(r['correct_answer']) == process_answer(r['agent_response'])
        is_correct_2 = False
        is_correct_3 = False
        try:
            is_correct_2 = process_answer(r['correct_answer_2']) == process_answer(r['agent_response'])
            is_correct_3 = process_answer(r['correct_answer_3']) == process_answer(r['agent_response'])
        except:
            pass

        return int(is_correct or is_correct_2 or is_correct_3)
        
    model_responses["is_correct"] = model_responses.apply(
        check_answer, axis=1
    )

    return model_responses

def prepare_unprocessed_response_dataframe(test_type):
    model_configs = read_model_configs()
    model_configs = [model_config for model_config in model_configs if model_config['model'] != 'Human']
    model_responses = create_all_agent_response_dataframe(
        test_name=test_type, 
        model_configs=model_configs,
        print_stats=False,
        prompt_type = "indist_instructions_question",
        response_file="model_responses"
    )

    # Add calvi standard to calvi trick
    if test_type == "calvi-trick":
        standard_model_responses = create_all_agent_response_dataframe(
            test_name="calvi-standard", 
            model_configs=model_configs,
            print_stats=False,
            prompt_type = "indist_instructions_question",
            response_file="model_responses"
        )
        standard_model_responses = standard_model_responses.replace({'calvi-standard': 'calvi-trick'})
        model_responses = pd.concat([model_responses, standard_model_responses]).reset_index()

    return model_responses

def prepare_dataframe(test_type):
    model_configs = read_model_configs()
    model_responses = create_all_agent_response_dataframe(
        test_name=test_type, 
        model_configs=model_configs,
        print_stats=False
    )
    if test_type in ['ggr', 'vlat', 'calvi-trick']:
        model_responses = find_correct_mc(model_responses_raw=model_responses)
        model_responses = model_responses.groupby(['agentType', 'question', 'image_file',])['is_correct'].mean().reset_index()
    else:
        model_responses = find_min_max_scaled_error(
            model_responses_raw=model_responses, 
            test_name=test_type,
            units_of_measure="is_correct"
        )
        model_responses = model_responses.dropna(subset=['is_correct'])
        model_responses = model_responses.groupby(['agentType', 'question', 'image_file',])['is_correct'].median().reset_index()
    
    model_responses = model_responses.sort_values(by=['question', 'image_file'])

    return model_responses

def prepare_response_dataframe(test_type):
    model_configs = read_model_configs()
    model_responses = create_all_agent_response_dataframe(
        test_name=test_type, 
        model_configs=model_configs,
        print_stats=False
    )
    if test_type in ['ggr', 'vlat', 'calvi-trick']:
        model_responses = find_correct_mc(model_responses_raw=model_responses)
    else:
        model_responses = find_min_max_scaled_error(
            model_responses_raw=model_responses, 
            test_name=test_type,
            units_of_measure="is_correct"
        )
        model_responses = model_responses.dropna(subset=['is_correct'])
    
    model_responses = model_responses.sort_values(by=['question', 'image_file'])

    return model_responses


def prepare_raw_response_dataframe(test_type):
    model_configs = read_model_configs()
    model_responses = create_all_agent_response_dataframe(
        test_name=test_type, 
        model_configs=model_configs,
        print_stats=False,
        response_file="model_responses"
    )
    model_responses = model_responses.sort_values(by=['question', 'image_file'])

    return model_responses

def prepare_original_raw_response_dataframe(test_type):
    model_configs = read_model_configs()
    model_responses = create_all_agent_response_dataframe(
        test_name=test_type, 
        model_configs=model_configs,
        print_stats=False,
        response_file="raw_model_responses"
    )
    model_responses = model_responses.sort_values(by=['question', 'image_file'])

    return model_responses

if __name__ == "__main__":
    all_raw_responses_df = []
    test_types = ['chartqa-test-continuous-human',  'ggr', 'vlat', 'holf', 'calvi-trick', 'holf2'] #'chartqa-test-continuous-human', 
    for test_type in test_types:
        print(test_type)
        model_response = prepare_original_raw_response_dataframe(test_type)
        model_response['test_type'] = test_type
        all_raw_responses_df.append(model_response)

    all_raw_responses_df = pd.concat(all_raw_responses_df)
    raw_responses_df = all_raw_responses_df[
        ['question', 'image_file', 'agentType', 'test_type', 'agent_response']
    ].value_counts().reset_index()

    # prepare_raw_response_dataframe('chartqa-test-continuous-human')
    # model_configs = read_model_configs()

    # for test_type in ['ggr', 'vlat', 'calvi-trick']:
    #     model_responses = create_all_agent_response_dataframe(
    #         test_name=test_type, 
    #         model_configs=model_configs,
    #         print_stats=True
    #     )
    #     model_responses = find_correct_mc(model_responses_raw=model_responses)
    #     agent_vectors = model_responses.groupby('agentType')['is_correct'].apply(list)
    #     agent_correlations = []
    #     for model_config_a in model_configs:
    #         agent_a = model_config_a['model']
    #         for model_config_b in model_configs:
    #             agent_b = model_config_b['model']
    #             agent_corr_a_b = pd.Series(agent_vectors[agent_a]).corr(
    #                 pd.Series(agent_vectors[agent_b])
    #             )
    #             agent_corr = {
    #                 'agent_a': agent_a,
    #                 'agent_b': agent_b,
    #                 'correlation': agent_corr_a_b
    #             }
    #             agent_correlations.append(agent_corr)
        
        # print(agent_correlations)


