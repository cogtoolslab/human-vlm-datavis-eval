import altair as alt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

alt.renderers.enable("mimetype")
alt.data_transformers.disable_max_rows()
rng = np.random.default_rng()

def bootstrap_ci(
        data: pd.DataFrame,
        measure: str,
        id_col: str,
        n_iterations=1000,
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
            "condition": condition,
            "ci_upper": upper, 
            "ci_lower": lower,
        })

    return pd.DataFrame(data_list)


def create_acc_plot(df: pd.DataFrame, measure: str):
    """Create 
    
    """

    agg_data = df.groupby(['condition']).agg(
        mean_prop=(measure, 'mean')
    ).reset_index()
    
    word_level_data = df.groupby(['condition', 'target_word']).agg(
        mean_prop=(measure, 'mean')
    ).reset_index()
    word_level_data['item_id'] = word_level_data['condition'] + word_level_data['target_word']

    ci_df = create_confidence_interval_df(
        data=df,
        measure=measure,
        id_col='target_word',
        condition_col='condition'
    )

    ci_plot = alt.Chart(ci_df).mark_errorbar().encode(
        x=alt.X("condition:N", title=None),
        y=alt.Y("ci_upper", title=None),
        y2=alt.Y2("ci_lower"),
        strokeWidth=alt.value(2),
        color=alt.Color('condition').legend(None)
    )
    
    scatter_plot = alt.Chart(word_level_data).mark_circle(size=16,opacity=0.5).encode(
        x=alt.X("condition:N", title=None),
        y=alt.Y("mean_prop:Q", scale=alt.Scale(domain=[0,1])),
        xOffset="jitter:Q",
        color=alt.Color('condition:N').legend(None),
    ).transform_calculate(
        jitter="sqrt(-2*log(random()))*cos(2*PI*random())" 
    )
    
    mean_points = alt.Chart(agg_data).mark_point(filled=True, size=50).encode(
        x=alt.X('condition:N'),
        y=alt.Y('mean_prop:Q', scale=alt.Scale(domain=[0,1])),
        color='condition:N'
    )

    return mean_points + scatter_plot + ci_plot