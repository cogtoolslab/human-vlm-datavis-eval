import pandas as pd


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


def create_all_agent_response_dataframe(
    test_name, 
    prompt_type="indist_instructions_question",
    model_configs=[],
    print_stats=False,
    response_file='processed_extracted_responses',
    dropna=True,
):
    AWS_PREFIX = "https://data-visualization-benchmark.s3.us-west-2.amazonaws.com"
    questions = pd.read_csv(f'{AWS_PREFIX}/{test_name}/questions.csv')

    model_responses = []
    for model_config in model_configs:
        top_p_dir = model_config['top_p']
        temperature_dir = model_config['temp']
        model = model_config['model'].replace("/", "-")

        _dir = f'responses/{prompt_type}/{top_p_dir}/{temperature_dir}'

        # all human results are store under processed_extracted_responses
        if model == "Human":
            current_response_file = "processed_extracted_responses"
        else:
            current_response_file = response_file

        # for get the chartqa model data from the normal dataset
        if test_name == "chartqa-test-continuous-human": # response_file == "model_responses" and 
            if model != "Human":
                test_type = "chartqa-test-continuous"
            else:
                test_type = "chartqa-test-continuous-human"
        else:
            test_type = test_name
        
        # if test_name == "chartqa-test-continuous-human":
        #     model_response['test_type'] = model_response['test_type'].replace(
        #         "chartqa-test-continuous", 
        #         "chartqa-test-continuous-human"
        #     )


        model_response_url = f'https://data-visualization-benchmark.s3.us-west-2.amazonaws.com/{test_type}/{_dir}/{model}/{current_response_file}.csv'
        model_response = pd.read_csv(model_response_url, low_memory=False)
        if "image_file" not in model_response.columns:
            model_response.rename(columns={
                "imageFile": "image_file",
            })

        if "agent_type" not in model_response.columns:
            model_response.rename(columns={
                "agentType": "agent_type",
            })

        if "test_type" not in model_response.columns:
            model_response.rename(columns={
                "testType": "test_type",
            })
            
        

        model_responses.append(model_response)

    model_responses = pd.concat(model_responses).reset_index(drop=True)
    model_responses = model_responses.merge(questions[['image_file','question','correct_answer']])
    if dropna:
        model_responses = model_responses.dropna(subset=["agent_response"])

    if print_stats:
        print(model_response_url)
        for model_config in model_configs:
            agent_type = model_config['model']
            agent_responses = model_responses[model_responses["agent_type"] == agent_type]
            print(f"Agent {agent_type} only responded to {len(agent_responses) / len(questions)} questions for test {test_type}")

    cols = ['image_file', 'question', 'correct_answer', 'agent_response', 'test_type', 'agent_type']
    
    # Do this mainly for human participants -- e.g. we need to do this for ggr
    if 'is_correct' in model_responses.columns:
        cols.append('is_correct')

    return model_responses[cols]

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
    # test_types = ['chartqa-test-continuous-human',  'ggr', 'vlat', 'holf', 'calvi-trick', 'holf2'] #'chartqa-test-continuous-human', 
    test_types = ['chartqa-test-continuous-human',]
    for test_type in test_types:
        print(test_type)
        model_response = prepare_original_raw_response_dataframe(test_type)
        model_response['test_type'] = test_type
        all_raw_responses_df.append(model_response)

    all_raw_responses_df = pd.concat(all_raw_responses_df)
    raw_responses_df = all_raw_responses_df[
        ['question', 'image_file', 'agent_type', 'test_type', 'agent_response']
    ].value_counts().reset_index()