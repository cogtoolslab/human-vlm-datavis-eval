import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numpy.random import Generator, PCG64
from utils import prepare_dataframe


def dom_bootstrap_ci(
    raw_data,
    col1,
    col2,
    n_iterations=1000,
    statistic=np.mean,
    units_of_measure="is_correct",
):
    
    data = raw_data.copy()
    data["question_image"] = data["question"] + " & " + data["image_file"]
    rng = Generator(PCG64())
    questions = list(data["question_image"].unique())
    n_size = len(questions)
    df = data.copy()

    def bootstrap_iteration(data, chosen_qs):
        filter_df = data[data["question_image"].isin(chosen_qs)] # Filter based on chosen questions
        col1_res = filter_df[(filter_df["agentType"] == col1)][units_of_measure]
        col2_res = filter_df[(filter_df["agentType"] == col2)][units_of_measure]
        
        bs_mean = statistic(col1_res) - statistic(col2_res)
        return bs_mean
        
    means = Parallel(n_jobs=-1)(
        delayed(bootstrap_iteration)(df, rng.choice(questions, n_size,  replace=True)) for _ in range(n_iterations)
    )
    
    # 95% confidence interval
    lower = np.percentile(means, 2.5)
    upper = np.percentile(means, 97.5)
    
    return lower, upper

def create_confidence_interval_dom_df(data, col1, col2, units_of_measure, statistic=np.mean):
    data_list = []
    lower, upper = dom_bootstrap_ci(
        data, 
        col1,
        col2,
        statistic=statistic, 
        units_of_measure=units_of_measure
    )

    data_list.append({
        # "agentType": agent,
        "ci_upper": upper, 
        "ci_lower": lower,
        "mean": (
            statistic(data[data["agentType"] == col1][units_of_measure]) - 
            statistic(data[data["agentType"] == col2][units_of_measure]))
    })

    return pd.DataFrame(data_list)

def create_all_dfs(agent_map, tests):
    all_dfs = []
    test_types = ['ggr', 'vlat', 'holf', 'calvi-trick', 'calvi-standard', 'holf2', 'chartqa-test-continuous-human']
    for test_type in test_types:
        model_response = prepare_dataframe(test_type)
        model_response['test_type'] = test_type
        all_dfs.append(model_response)

    all_dfs = pd.concat(all_dfs).reset_index()
    all_dfs['agentType'] = all_dfs['agentType'].replace(agent_map)
    all_dfs = all_dfs[all_dfs['test_type'].isin(tests)]
    return all_dfs

def create_ci_df(agent_map, item_type, col1, col2):
    if item_type == "multiple_choice":
        tests = ['ggr', 'vlat',' calvi-trick', 'calvi-standard']
        statistic = np.mean
    else:
        tests = ['holf','holf2','chartqa-test-continuous-human']
        statistic = np.median

    all_dfs = create_all_dfs(agent_map=agent_map, tests=tests)
    create_confidence_interval_dom_df(
        all_dfs, 
        col1=col1, 
        col2=col2, 
        units_of_measure='is_correct',
        statistic=statistic
    )


if __name__ == "__main__":
    agent_map = {
        "llava-hf/llava-1.5-7b-hf": np.nan,
        "llava-hf/llava-1.5-13b-hf":  np.nan,
        "liuhaotian/llava-v1.6-34b": "model",
        'Salesforce/blip2-flan-t5-xl':  np.nan,
        'Salesforce/blip2-flan-t5-xxl':  np.nan,
        "google/pix2struct-chartqa-base":  np.nan,
        "google/matcha-chartqa":  np.nan,
        'GPT-4V':  np.nan,
        'Human': 'human',
    }
    create_ci_df(agent_map=agent_map, 
                 item_type="multiple_choice",
                 col1="human",
                 col2="model")
    
    