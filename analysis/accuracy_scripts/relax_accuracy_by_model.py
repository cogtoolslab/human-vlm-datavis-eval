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
        filter_df = data[data[id_col].isin(chosen_items)] # Filter based on chosen questions
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
        

def create_acc_plot(
        df: pd.DataFrame, 
        measure: str, 
        x_domain: List[str], 
        main_colors: List[str], 
        accent_colors: List[str]
    ):
    """Create 
    """

    agg_data_df = []
    for relax_accuracy_int in range(0, 6, 1):
        relax_accuracy_alpha = relax_accuracy_int / 10
        relaxed_accuracy_df = df.copy()
        relaxed_accuracy_df['is_correct'] = relaxed_accuracy_df.apply(
            lambda r: get_relaxed_accuracy(r['agent_response'], r['correct_answer'], relax_accuracy_alpha), 
            axis=1
        )

        agg_data = relaxed_accuracy_df.groupby(['agent_type']).agg(
            mean_prop=(measure, 'mean')
        ).reset_index()

        agg_data['relax_accuracy_alpha'] = relax_accuracy_alpha
        agg_data_df.append(agg_data)
        

    agg_data_df = pd.concat(agg_data_df)
    ci_df = agg_data_df.groupby('agent_type').agg(
        ci_lower=('mean_prop', 'min'),
        ci_upper=('mean_prop', 'max')
    ).reset_index()

    ci_plot = alt.Chart(ci_df).mark_errorbar().encode(
        x=alt.X("agent_type", title=None),
        y=alt.Y("ci_upper", title=None),
        y2=alt.Y2("ci_lower"),
        strokeWidth=alt.value(2),
        color=alt.Color('agent_type', scale=alt.Scale(domain=x_domain, range=main_colors)).legend(None)
    )
    
    mean_points = alt.Chart(agg_data_df).mark_point(filled=True, size=50).encode(
        x=alt.X('agent_type', scale=alt.Scale(domain=x_domain)),
        y=alt.Y('mean_prop:Q', scale=alt.Scale(domain=[0,1]), title='Relaxed Accuracy'),
        color=alt.Color('agent_type', scale=alt.Scale(domain=x_domain, range=accent_colors)),
        opacity=alt.Opacity('relax_accuracy_alpha:N',)
    )

    return ci_plot + mean_points

if __name__ == "__main__":
    df = pd.read_csv("/Users/arnav/Desktop/human-vlm-datavis-eval/analysis/all_valid_responses_df.csv", low_memory=False)
    df['item_id'] = df['question'] + df['image_file']

    configs = read_model_configs()
    main_colors = [config['color'] for config in configs]
    accent_colors = [config['accent_color'] for config in configs]
    domain_agent_order = [config['model'] for config in configs]
    
    for test in ['holf', 'holf2', 'chartqa-test-continuous-human']:
        test_df = df[df['test_type'] == test].copy()
        test_df['agent_response'] = test_df['agent_response'].astype(float)
        test_df['correct_answer'] = test_df['correct_answer'].astype(float)
        plot = create_acc_plot(
            test_df, 
            measure='is_correct', 
            main_colors=main_colors, 
            accent_colors=accent_colors, 
            x_domain=domain_agent_order
        ).properties(title=f"{test}")

        plot.save(f"./accuracy_plot/accuracy_{test}.pdf")