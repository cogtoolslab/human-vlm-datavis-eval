import pandas as pd
import altair as alt
import numpy as np
import math
from scipy import stats
from utils import prepare_dataframe, read_model_configs

def create_human_response_dfs():
    human_response_dfs = []

    for test_type in ['ggr', 'vlat', 'calvi-trick', 'calvi-standard']:
        file = "processed_extracted_responses.csv"
        if test_type in ['ggr', 'vlat']:
            file = "processed_extracted_responses_ids.csv"
        url = f"https://data-visualization-benchmark.s3.us-west-2.amazonaws.com/{test_type}/responses/indist_instructions_question/pna/tna/Human/"
        human_response_df = pd.read_csv(url + file)
        human_response_df['test_type'] = test_type
        human_response_dfs.append(human_response_df)

    human_response_dfs = pd.concat(human_response_dfs).reset_index().replace({
        'calvi-standard': 'calvi-trick'
    })
    return human_response_dfs

def get_participant_split(human_response_dfs, test_type):
    test_participants = human_response_dfs[
        human_response_dfs["test_type"] == test_type
    ][["participant_id"]].sample(frac=1, random_state=50)
    test_participants = human_response_dfs[["participant_id"]].drop_duplicates().sample(frac=1, random_state=50)
    
    half_idx = (len(test_participants) // 2)
    half1 = test_participants.iloc[:half_idx]['participant_id'].to_list()
    half2 = test_participants.iloc[half_idx:]['participant_id'].to_list()
    
    return half1, half2

def get_participant_average(df, participants):
    response_df = df[df['participant_id'].isin(participants)]
    response_df = response_df.groupby(['question', 'image_file', 'test_type'])['is_correct'].mean()
    return response_df.reset_index().sort_values(by=['question','image_file'])


def get_corrected_corr(df1, df2, split_half=True):
    correlation_result = stats.pearsonr(df1, df2)
    correlation = correlation_result.statistic
    correlation_ci_low = correlation_result.confidence_interval().low
    correlation_ci_high = correlation_result.confidence_interval().high

    corrected_corr = [correlation, correlation_ci_low, correlation_ci_high]

    if split_half:
        # https://en.wikipedia.org/wiki/Spearman%E2%80%93Brown_prediction_formula
        spearman_brown_correction = lambda correlation : 2 * correlation / (1 + correlation)
        corrected_corr = [spearman_brown_correction(corr) for corr in corrected_corr]
    return corrected_corr

def find_splithalf_reliability(human_response_dfs):
    participants_1 = []
    participants_2 = []
    test_types = ['ggr', 'vlat',  'calvi-trick',]
    for test_type in test_types:
        half1, half2 = get_participant_split(human_response_dfs, test_type)
        participants_1 += half1
        participants_2 += half2

    human_response_dfs_1 = get_participant_average(human_response_dfs, participants_1)
    human_response_dfs_2 = get_participant_average(human_response_dfs, participants_2)
    corrected_corr  = get_corrected_corr(human_response_dfs_1['is_correct'], human_response_dfs_2['is_correct']) 
    corrected_corr.append('Human')

    test_correlations = []
    for test in test_types:
            test_human1_df = human_response_dfs_1[human_response_dfs_1['test_type'] == test]
            test_human2_df = human_response_dfs_2[human_response_dfs_2['test_type'] == test]
            test_corr = get_corrected_corr(test_human1_df['is_correct'], test_human2_df['is_correct'], split_half=True)
            test_correlations.append(test_corr + ['Human', test])

    return corrected_corr, test_correlations

def create_model_response_df():
    
    all_responses_df = []
    test_types = ['ggr', 'vlat', 'calvi-trick', 'calvi-standard']
    for test_type in test_types:
        model_response = prepare_dataframe(test_type)
        model_response['test_type'] = test_type
        all_responses_df.append(model_response)

    all_responses_df = pd.concat(all_responses_df)
    return all_responses_df

def find_model_model_correlation():
    models = [model_config['model'] for model_config in read_model_configs()]
    model_df = create_model_response_df()
    model_df['item'] = model_df['question'] + " & " + model_df['image_file']
    all_correlations = []

    for model in models:
        for model2 in models:
            if model == "Human" or model2 == "Human":
                continue
            current_model2_df = model_df[model_df['agentType'] == model2].copy()
            current_model_df = model_df[model_df['agentType'] == model].copy()

            current_model_df = current_model_df[current_model_df['item'].isin(current_model2_df['item'].unique())]
            current_model2_df = current_model2_df[current_model2_df['item'].isin(current_model_df['item'].unique())]

            corrected_corr = get_corrected_corr(current_model_df['is_correct'], current_model2_df['is_correct'], split_half=False)
            all_correlations.append(corrected_corr + [model, model2])

    return all_correlations

def find_human_model_correlation():
    models = [model_config['model'] for model_config in read_model_configs()]
    test_types = ['ggr', 'vlat', 'calvi-trick',]

    model_df = create_model_response_df()
    model_df['item'] = model_df['question'] + " & " + model_df['image_file']
    human_df = model_df[model_df['agentType'] == 'Human'].copy()
    all_correlations = []
    test_correlations = []

    for model in models:
        if model == "Human":
            continue
        current_model_df = model_df[model_df['agentType'] == model]
        current_human_df = human_df[human_df['item'].isin(current_model_df['item'])]

        corrected_corr = get_corrected_corr(current_model_df['is_correct'], current_human_df['is_correct'], split_half=False)

        for test in test_types:
            test_model_df = current_model_df[current_model_df['test_type'] == test]
            test_human_df = current_human_df[current_human_df['test_type'] == test]
            test_corr = get_corrected_corr(test_model_df['is_correct'], test_human_df['is_correct'], split_half=False)
            test_correlations.append(test_corr + [model, test])

        all_correlations.append(corrected_corr + [model])

    return all_correlations, test_correlations


def create_ci_plot(correlation, correlation_tests):
    model_configs = read_model_configs()
    model_order = [model_config['model'] for model_config in model_configs]
    color_order = [model_config['color'] for model_config in model_configs]


    # Create the plot
    ci_chart = alt.Chart(correlation).mark_circle(
        size=40, 
        filled=True,
        strokeWidth=1,
        opacity=1
    ).encode(
        x=alt.X(
            'model_compared:N',
            scale=alt.Scale(
                domain=model_order,
            )
        ),
        y='correlation:Q',
        color=alt.Color(
            'model_compared',
            scale=alt.Scale(
                domain=model_order,
                range=color_order)
        )
    )

    ci_error_bars = ci_chart.mark_errorbar().encode(
        y='lower:Q',
        y2='upper:Q',
    )

    scatter = alt.Chart(correlation_tests).mark_circle(
        size=8, opacity=0.6
    ).encode(
        x=alt.X('model_compared:N', scale=alt.Scale(domain=model_order)),
        y='correlation:Q',
        xOffset="jitter:Q",
        color=alt.Color(
            'model_compared:N',
            scale=alt.Scale(domain=model_order, range=color_order),
        ),
    ).transform_calculate(
        jitter="sqrt(-2*log(random()))*cos(2*PI*random())" 
    )

    chart = scatter + ci_chart + ci_error_bars
    chart = chart.properties(title="Correlation to average human response", height=300, width=125)
    return chart


if __name__ == "__main__":
    human_response_dfs = create_human_response_dfs()
    human_split_half_corr, human_split_test_correlations = find_splithalf_reliability(human_response_dfs)
    human_model_corr, human_model_test_corr = find_human_model_correlation()

    human_model_corr.append(human_split_half_corr)
    human_model_test_corr += human_split_test_correlations

    correlation_df = pd.DataFrame(human_model_corr, columns=['correlation', 'lower', 'upper', 'model_compared'])
    correlation_df.to_csv("./correlation/human_model_correlation_df.csv")

    human_model_test_df = pd.DataFrame(human_model_test_corr, columns=['correlation', 'lower', 'upper', 'model_compared', 'test'])
    human_model_test_df.to_csv("./correlation/human_model_test_df.csv")

    model_model_corr = find_model_model_correlation()
    model_model_corr_df = pd.DataFrame(model_model_corr, columns=['correlation', 'lower', 'upper', 'model_compared', 'model2_compared'])
    model_model_corr_df.to_csv("./correlation/model_model_corr_df.csv")

    plot_file = "./correlation/human_model_correlation_ci.pdf"
    create_ci_plot(correlation_df, human_model_test_df).save(plot_file)
    print(plot_file)
