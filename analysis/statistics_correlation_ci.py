from create_correlation_heatmap import create_correlation_matrix
import pandas as pd
import altair as alt
from utils import read_model_configs, create_all_agent_response_dataframe, get_all_test_types, prepare_dataframe, prepare_response_dataframe
from utils import find_correct_mc, find_min_max_scaled_error
import numpy as np
import math
from joblib import Parallel, delayed
from numpy.random import Generator, PCG64
alt.renderers.enable("mimetype")
alt.data_transformers.disable_max_rows()

def create_response_dfs():
    all_response_dfs = []
    test_types = ['ggr', 'vlat', 'calvi-trick']
    for test_type in test_types:
        model_response = prepare_response_dataframe(test_type)
        model_response['test_type'] = test_type
        all_response_dfs.append(model_response)

    all_response_dfs = pd.concat(all_response_dfs)
    return all_response_dfs

def prepare_dataframe_corr(test_type):
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

def correlation_bootstrap_ci(
    raw_data,
    model_configs,
    n_iterations=1000,
):
    
    data = raw_data.copy()
    data["question_image"] = data["question"] + " & " + data["image_file"]
    rng = Generator(PCG64())
    questions = list(data["question_image"].unique())
    n_size = len(questions)
    df = data.copy()

    def bootstrap_iteration(data, chosen_qs):
        filter_df = data[data["question_image"].isin(chosen_qs)] # Filter based on chosen questions
        corr_matrix = create_correlation_matrix(filter_df, model_configs=model_configs)
        return corr_matrix['correlation'].iloc[1]
        
    means = Parallel(n_jobs=-1)(
        delayed(bootstrap_iteration)(df, rng.choice(questions, n_size,  replace=True)) for _ in range(n_iterations)
    )
    
    # 95% confidence interval
    lower = np.percentile(means, 2.5)
    upper = np.percentile(means, 97.5)
    mean_corr = create_correlation_matrix(raw_data, model_configs=model_configs)['correlation'].iloc[1]
    
    return {
        "ci_upper": upper, 
        "ci_lower": lower,
        "mean": mean_corr
    }

def correlation_ci(agent_map):
    model_responses = create_response_dfs()
    human_model_error = model_responses.copy()
    human_model_error['agentType'] = human_model_error['agentType'].replace(agent_map)
    human_model_error = human_model_error.dropna(subset=['agentType'])
    human_model_error = human_model_error.groupby(['agentType', 'question', 'image_file',])['is_correct'].mean().reset_index()
    model_configs = [{'model': agent} for agent in human_model_error['agentType'].unique() if agent != np.nan]
    ci_df = correlation_bootstrap_ci(
        human_model_error,
        model_configs=model_configs,
        n_iterations=1000,
    )
    return ci_df

if __name__ == "__main__":
    agent_map = {
        "llava-hf/llava-1.5-7b-hf": np.nan,
        "llava-hf/llava-1.5-13b-hf": np.nan,
        "liuhaotian/llava-v1.6-34b": np.nan,
        'Salesforce/blip2-flan-t5-xl': np.nan,
        'Salesforce/blip2-flan-t5-xxl': np.nan,
        "google/pix2struct-chartqa-base": np.nan,
        "google/matcha-chartqa": np.nan,
        'GPT-4V': 'model',
        'Human': "Humans",
    }
    print(correlation_ci(agent_map))