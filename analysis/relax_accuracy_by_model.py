import altair as alt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from base_utils import read_model_configs, get_relaxed_accuracy
from typing import List
import re

alt.renderers.enable("mimetype")
alt.data_transformers.disable_max_rows()
rng = np.random.default_rng()

def bootstrap_ci(
        data: pd.DataFrame,
        measure: str,
        id_col: str,
        n_iterations=10,
        statistic=np.mean):
    """Bootstrap confidence interval
    """
    
    items = list(data[id_col].unique())
    n_size = len(items)
    df = data.copy()

    def bootstrap_iteration(data, chosen_items):
        filter_df = pd.concat(
            [data[data['item_id'] == item] for item in chosen_items]
        )
        bs_mean = statistic(filter_df[measure]) 
        return (bs_mean, list(chosen_items))

    qset_means = Parallel(n_jobs=-1)(
        delayed(bootstrap_iteration)(
            df.copy(),
            rng.choice(items, n_size,  replace=True)
        ) for _ in range(n_iterations)
    )
    
    means = []
    qs_used = []
    means = [bs_mean for bs_mean, chosen_qs in qset_means]
 
    # 95% confidence interval
    lower = np.percentile(means, 2.5)
    upper = np.percentile(means, 97.5)
    
    return lower, upper


def create_confidence_interval_df(
    data: pd.DataFrame,
    measure: str, 
    id_col: str,
    condition_col: str,
    statistic=np.mean
):
    data_list = []

    for condition in data[condition_col].unique():
        condition_data = data[data[condition_col] == condition]

        lower, upper = bootstrap_ci(condition_data, measure=measure, statistic=statistic, id_col=id_col)

        data_list.append({
            condition_col: condition,
            "ci_upper": upper, 
            "ci_lower": lower,
        })

    return pd.DataFrame(data_list)

def ensure_item_count(df: pd.DataFrame, target_count: int = 10):
    """
    Ensure each agent has exactly `target_count` rows for every item_id.
    Missing rows will have NaN for the "is_correct" column.
    
    Parameters:
    - df (pd.DataFrame): Input DataFrame with columns "item_id", "is_correct", "agent_id".
    - target_count (int): The number of rows required for each item_id per agent.
    
    Returns:
    - pd.DataFrame: Modified DataFrame with missing rows added.
    """
    all_item_ids = set(df["item_id"].unique())
    result = []
    
    for agent, group in df.groupby("agent_type"):
        existing_item_ids = set(group["item_id"])
        missing_item_ids = all_item_ids - existing_item_ids
        
        # Add rows for existing item_ids
        for item_id in all_item_ids:
            item_rows = group[group["item_id"] == item_id]
            missing_count = target_count - len(item_rows)
            
            # Add existing rows
            result.append(item_rows)
            
            # Add missing rows if needed
            if missing_count > 0:
                result.append(pd.DataFrame({
                    "item_id": [item_id] * missing_count,
                    "is_correct": [np.nan] * missing_count,
                    "agent_type": [agent] * missing_count
                }))
        
        # Add rows for completely missing item_ids
        for missing_item in missing_item_ids:
            result.append(pd.DataFrame({
                "item_id": [missing_item] * target_count,
                "is_correct": [np.nan] * target_count,
                "agent_type": [agent] * target_count
            }))
    
    return pd.concat(result).reset_index(drop=True)
        

def create_acc_plot(
        df: pd.DataFrame, 
        measure: str, 
        x_domain: List[str], 
        main_colors: List[str], 
        accent_colors: List[str],
        show_recall=True
    ):
    """Create 
    """

    agg_data_df = []
    for relax_accuracy_int in range(0, 6, 1):
        relax_accuracy_alpha = relax_accuracy_int / 10
        relaxed_accuracy_df = df[df['group_includes'] == 'valid'].copy()
        relaxed_accuracy_df['is_correct'] = relaxed_accuracy_df.apply(
            lambda r: get_relaxed_accuracy(r['agent_response'], r['correct_answer'], relax_accuracy_alpha), 
            axis=1
        )

        agg_data = relaxed_accuracy_df.groupby(['agent_type']).agg(
            mean_prop=(measure, 'mean')
        ).reset_index()

        agg_data['relax_accuracy_alpha'] = relax_accuracy_alpha
        agg_data['data_type'] = 'valid_only'
        agg_data_df.append(agg_data)

        # accuracy df with 10 items for agent
        # complete_accuracy_df = ensure_item_count(relaxed_accuracy_df, target_count=10)
        complete_accuracy_df = df[df['group_includes'] == 'all'].copy()
        # complete_accuracy_df = ensure_item_count(complete_accuracy_df, target_count=10)
        complete_accuracy_df['is_correct'] = complete_accuracy_df.apply(
            lambda r: get_relaxed_accuracy(r['agent_response'], r['correct_answer'], relax_accuracy_alpha), 
            axis=1
        )
        # complete_accuracy_df = complete_accuracy_df.fillna(0)
        
        complete_agg_data = complete_accuracy_df.groupby(['agent_type']).agg(
            mean_prop=(measure, 'mean')
        ).reset_index()
        complete_agg_data['relax_accuracy_alpha'] = relax_accuracy_alpha
        complete_agg_data['data_type'] = 'full'
        complete_agg_data = complete_agg_data[complete_agg_data['agent_type'] != 'Human']
        agg_data_df.append(complete_agg_data)



    agg_data_df = pd.concat(agg_data_df)
    ci_df = agg_data_df.groupby(['agent_type', 'data_type']).agg(
        ci_lower=('mean_prop', 'min'),
        ci_upper=('mean_prop', 'max')
    ).reset_index()

    ci_plot = alt.Chart(ci_df).mark_errorbar(opacity=0.8).encode(
        x=alt.X("agent_type", title=None),
        y=alt.Y("ci_upper", title=None),
        y2=alt.Y2("ci_lower"),
        strokeWidth=alt.value(2),
        color=alt.Color('agent_type', scale=alt.Scale(domain=x_domain, range=accent_colors)).legend(None),
        xOffset="data_type:N",
    )
    
    mean_points = alt.Chart(agg_data_df).mark_point(filled=True, size=30, strokeWidth=0.5).encode(
        x=alt.X('agent_type', scale=alt.Scale(domain=x_domain)),
        y=alt.Y('mean_prop:Q', scale=alt.Scale(domain=[0,1]), title='Relaxed Accuracy'),
        opacity=alt.Opacity('relax_accuracy_alpha:N').legend(None),
        # color=alt.Color('agent_type', scale=alt.Scale(domain=x_domain, range=accent_colors)),
        color=alt.condition(
            alt.datum['data_type'] == 'full',
            alt.value('white'),
            alt.Color('agent_type', scale=alt.Scale(domain=x_domain, range=main_colors))
        ).legend(None),
        xOffset="data_type",
        stroke=alt.Stroke('agent_type', scale=alt.Scale(domain=x_domain, range=main_colors)).legend(None),
    )

    return ci_plot + mean_points


def process_model_responses(model_configs, test_name):
    AWS_PREFIX = "https://data-visualization-benchmark.s3.us-west-2.amazonaws.com"
    prompt_type = "indist_instructions_question"
    response_file ='processed_extracted_responses'
    
    questions = pd.read_csv(f'{AWS_PREFIX}/{test_name}/questions.csv')

    if test_name == "calvi-trick":
        calvi_standard_questions = pd.read_csv(f'{AWS_PREFIX}/calvi-standard/questions.csv')
        questions = pd.concat([questions, calvi_standard_questions])

    model_responses = []
    for model_config in model_configs:
        top_p_dir = model_config['top_p']
        temperature_dir = model_config['temp']
        model = model_config['model'].replace("/", "-")

        _dir = f'responses/{prompt_type}/{top_p_dir}/{temperature_dir}'
        model_response_url = f'{AWS_PREFIX}/{test_name}/{_dir}/{model}/{response_file}.csv'
        
        # print(model_response_url)
        model_response = pd.read_csv(model_response_url, low_memory=False)
        model_response['agent_type'] = model_config['model']

        if test_name == "calvi-trick":
            def get_calvi_standard():
                standard_test_name = "calvi-standard"
                standard_response_url = f'{AWS_PREFIX}/{standard_test_name}/{_dir}/{model}/{response_file}.csv'
                standard_response = pd.read_csv(standard_response_url)
                standard_response['testType'] = standard_response['testType'].replace("calvi-standard", "calvi-trick")
                return standard_response

            calvi_standard_df = get_calvi_standard()
            calvi_df = pd.concat([calvi_standard_df, model_response]).reset_index(drop=True)
            model_responses.append(calvi_df)
        else:
            model_responses.append(model_response)

    model_responses = pd.concat(model_responses).reset_index(drop=True)

    if test_name != "calvi-trick":
        model_responses = model_responses[model_responses["testType"] == test_name]

    if 'image_file' not in model_responses.columns:
        model_responses = model_responses.rename({"imageFile": "image_file"}, axis=1)

    if 'agentType' not in model_responses.columns:
        model_responses = model_responses.rename({"agent_type": "agentType"}, axis=1)

    model_responses = model_responses.drop(['correct_answer'], axis=1)
    model_responses = model_responses.merge(questions[['image_file', 'question', 'correct_answer']])
    model_responses['group_includes'] = 'all'

    model_responses_valid = model_responses.dropna(subset=["agent_response"]).copy()
    model_responses_valid['group_includes'] = 'valid'

    model_responses = pd.concat([model_responses, model_responses_valid])

    return model_responses

    

if __name__ == "__main__":
    df = pd.read_csv("/Users/arnav/Desktop/human-vlm-datavis-eval/analysis/all_valid_responses_df.csv", low_memory=False)
    df['item_id'] = df['question'] + df['image_file']

    configs = read_model_configs()
    main_colors = [config['color'] for config in configs]
    accent_colors = [config['accent_color'] for config in configs]
    domain_agent_order = [config['model'] for config in configs]
    
    plots = []
    for test in ['holf', 'holf2', 'chartqa-test-continuous-human']: #
        # test_df = process_model_responses(configs, test_name=test)
        prefix = "/Users/arnav/Desktop/human-vlm-datavis-eval/analysis"
        test_df = pd.read_csv(prefix + f"/single_accuracy_plots/{test}_invalid_included_df.csv", low_memory=False)
        test_df['item_id'] = test_df['question'] + test_df['image_file']
        test_df['agent_type'] = test_df['agentType']
        
        # test_df = df[df['test_type'] == test].copy()
    
        test_df['agent_response'] = test_df['agent_response'].astype(float)
        test_df['correct_answer'] = test_df['correct_answer'].astype(float)
        plot = create_acc_plot(
            test_df, 
            measure='is_correct', 
            main_colors=main_colors, 
            accent_colors=accent_colors, 
            x_domain=domain_agent_order
        ).properties(title=f"{test}", width=125)
        plots.append(plot)

    alt.hconcat(*plots).save(f"./accuracy_plot/relaxed_accuracy.pdf")
    # plot.save(f"./accuracy_plot/accuracy_{test}.pdf")