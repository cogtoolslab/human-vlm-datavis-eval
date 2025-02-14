import pandas as pd
import altair as alt
from base_utils import read_model_configs, create_all_agent_response_dataframe, find_correct_mc, find_min_max_scaled_error
import numpy as np
import math
from base_utils import get_relaxed_accuracy



def prepare_dataframe(test_type):
    model_configs = read_model_configs()
    model_responses = create_all_agent_response_dataframe(
        test_name=test_type, 
        model_configs=model_configs,
        print_stats=True
    )
    if test_type in ['ggr', 'vlat', 'calvi-trick', 'calvi-standard']:
        model_responses = find_correct_mc(model_responses_raw=model_responses)
        model_responses = model_responses.groupby(['agent_type', 'question', 'image_file',])['is_correct'].mean().reset_index()
    else:
        # model_responses = find_min_max_scaled_error(
        #     model_responses_raw=model_responses, 
        #     test_name=test_type,
        #     units_of_measure="is_correct"
        # )
        # model_responses = model_responses.dropna(subset=['is_correct'])
        model_responses['is_correct'] = model_responses.apply(
            lambda r : get_relaxed_accuracy(r["agent_response"], r["correct_answer"], 0.05),
            axis=1
        )
        model_responses = model_responses.groupby(['agent_type', 'question', 'image_file',])['is_correct'].median().reset_index()

    # model_responses = model_responses[model_responses['is_correct'] < 1.6]
    # model_responses = model_responses.dropna(subset=['agent_response'])
    
    model_responses = model_responses.sort_values(by=['question', 'image_file'])
    
    # = model_responses.loc[model_responses.groupby(['agent_type', 'question', 'image_file',])['is_correct'].idxmin()]

    return model_responses

def create_correlation_matrix(model_responses,model_configs):
    
    # agent_vectors = model_responses.groupby('agent_type')['is_correct'].apply(list)
    # items = model_responses[['question', 'image_file']].drop_duplicates()
    # cooks_d = pd.read_csv("./cooks_distance/cooks_distance.csv")
    # cooks_d = cooks_d[cooks_d['cooks_distance'] <= 1]

    agent_correlations = []
    for model_config_a in model_configs:
        agent_a = model_config_a['model']
        for model_config_b in model_configs:
            agent_b = model_config_b['model']

            # agents_cooks_d = cooks_d[ 
            #     (cooks_d['agent_a'] == agent_a) &
            #     (cooks_d['agent_b'] == agent_b)
            # ].copy()
            # agents_cooks_d['item'] = agents_cooks_d['question'] + " & " + agents_cooks_d['image_file']

            responses_a = model_responses[model_responses['agent_type'] == agent_a]
            responses_b = model_responses[model_responses['agent_type'] == agent_b]
            

            responses = responses_a.merge(responses_b, on=['question', 'image_file'],  suffixes=('_a', '_b'))
            
            responses['item'] = responses['question'] + " & " + responses['image_file']
            # responses = responses[responses['item'].isin(agents_cooks_d['item'])]

            responses = responses.dropna(subset=['is_correct_a', 'is_correct_b'])

            agent_vector_a = responses['is_correct_a']
            agent_vector_b = responses['is_correct_b'] 

            agent_corr_a_b = np.corrcoef(agent_vector_a, agent_vector_b)[1][0]
            # print(len(responses_a), agent_a, len(responses_b), agent_b, agent_corr_a_b)
            if math.isnan(agent_corr_a_b):
                pass
            # if agent_a == "llava-hf/llava-1.5-7b-hf" and agent_b == "Salesforce/blip2-flan-t5-xl":
            #     pass

            agent_corr = {
                'agent_type_1': agent_a,
                'agent_type_2': agent_b,
                'correlation': agent_corr_a_b
            }
            agent_correlations.append(agent_corr)

    return pd.DataFrame(agent_correlations)

def create_pairwise_agent_heatmap(df, response_type, units_of_measure):
    model_config = read_model_configs()

    base = alt.Chart(df).mark_rect().encode(
        x=alt.X("agent_type_1", scale=alt.Scale(domain=[model['model'] for model in model_config])),  
        y=alt.Y("agent_type_2", scale=alt.Scale(domain=[model['model'] for model in model_config])),
    )

    if response_type == 'multiple_choice':
        color_domain=[-1,1]
        color_reverse = False
        color_condition = alt.condition(
            alt.datum[units_of_measure] < 0.65,
            alt.value('black'),
            alt.value('white')
        )
    else:
        color_domain=[0,1.6]
        color_reverse = True
        color_condition = alt.condition(
            alt.datum[units_of_measure] > 0.65,
            alt.value('black'),
            alt.value('white')
        )

    height=300
    width=400 
    title=f"Numerical Items Accuracy Vector Correlation between Agents"

    heatmap = base.mark_rect().encode(
        color=alt.Color(f'{units_of_measure}:Q', 
                        legend=None, 
                        scale=alt.Scale(
                            # scheme="blues", 
                            scheme="redgrey",
                            domain=color_domain,
                            reverse=color_reverse)),
    )

    text = base.mark_text(baseline='middle').encode(
        alt.Text(f'{units_of_measure}:Q', format=".2f"),
        color=color_condition
    )

    chart = heatmap.properties(
        width=width,
        height=height,
        title=title
    ) + text

    return chart

if __name__ == '__main__':
    model_responses = []
    # ['ggr', 'vlat', 'calvi-trick']
    # for test_type in ['ggr', 'vlat', 'calvi-trick', 'calvi-standard']:#['holf', 'chartqa-test-continuous-human']:
    for test_type in ['holf', 'holf2',]:
        model_response = prepare_dataframe(test_type)
        model_responses.append(model_response)

    model_responses = pd.concat(model_responses).replace({"calvi-trick": "calvi", "calvi-standard": "calvi"})
    # model_responses['agent_type'] = model_responses['agent_type'].copy()
    # model_responses['test_type'] = model_responses['test_type'].copy()

    model_responses_file = "./heatmap/model_responses.csv"
    print("MATRIX FILE: ", model_responses_file)
    model_responses.to_csv(model_responses_file)
    df = create_correlation_matrix(model_responses, model_configs=read_model_configs())
    matrix_file_name = "./heatmap/correlation_matrix.csv"
    print("MATRIX FILE: ", matrix_file_name)
    df.to_csv(matrix_file_name)
    test_type = "multiple_choice"
    heatmap = create_pairwise_agent_heatmap(df, test_type, 'correlation')
    filename = f"./heatmap/between_model_corr_mc.pdf"
    print("Saving...", filename)
    heatmap.save(filename)