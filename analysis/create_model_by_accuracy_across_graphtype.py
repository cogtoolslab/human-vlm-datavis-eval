import altair as alt
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from numpy.random import Generator, PCG64
alt.renderers.enable("mimetype")
alt.data_transformers.disable_max_rows()
import os
from evaluation_metrics import EvaluationMetrics

class AccuracyPlot:

    def __init__(
        self, 
        test_names, 
        prompt_type, 
        units_of_measure = 'a_in_b', 
        model_configs=[],
        print_stats=False,
        response_file='processed_extracted_responses'
    ):
        self.model_configs = model_configs
        self.AWS_PREFIX = "https://data-visualization-benchmark.s3.us-west-2.amazonaws.com"
        self.evaluation_metric = EvaluationMetrics()
        self.prompt_type = prompt_type

        all_responses = []
        self.test_names = test_names
        for test_name in test_names:
            self.test_name = test_name
            self.questions = pd.read_csv(f'{self.AWS_PREFIX}/{test_name}/questions.csv')
            if self.test_name == "calvi-trick": # read all questions, not just calvi trick
                calvi_standard_questions = pd.read_csv(f'{self.AWS_PREFIX}/calvi-standard/questions.csv')
                self.questions = pd.concat([self.questions, calvi_standard_questions])
            
            
            self.units_of_measure = units_of_measure

            self.model_responses = []
            for model_config in model_configs:
                top_p_dir = model_config['top_p']
                temperature_dir = model_config['temp']
                model = model_config['model'].replace("/", "-")

                _dir = f'responses/{prompt_type}/{top_p_dir}/{temperature_dir}'
                model_response_url = f'https://data-visualization-benchmark.s3.us-west-2.amazonaws.com/{self.test_name}/{_dir}/{model}/{response_file}.csv'
            
                print(model_response_url)
                model_response = pd.read_csv(model_response_url, low_memory=False)
                
                
                # combine calvi-trick to be a part of calvi-standard
                if self.test_name == "calvi-trick":
                    def get_calvi_standard():
                        testName = "calvi-standard"
                        model_response_url = f'https://data-visualization-benchmark.s3.us-west-2.amazonaws.com/{testName}/{_dir}/{model}/{response_file}.csv'
                        model_response = pd.read_csv(model_response_url)
                        # model_response['testType'] = model_response['test_type']
                        model_response['testType'] = model_response['testType'].replace("calvi-standard", "calvi-trick")
                        return model_response
                    calvi_standard_df = get_calvi_standard()
                    calvi_df = pd.concat([calvi_standard_df, model_response]).reset_index(drop=True)
                    self.model_responses.append(calvi_df)
                else:
                    self.model_responses.append(model_response)

            self.model_responses = pd.concat(self.model_responses).reset_index(drop=True)
            


            if self.test_name != "calvi-trick":
                self.model_responses = self.model_responses[self.model_responses["testType"] == self.test_name]
            
            if 'image_file' not in self.model_responses.columns:
                self.model_responses = self.model_responses.rename({
                    "imageFile": "image_file"
                }, axis=1)
            if 'agentType' not in self.model_responses:
                self.model_responses = self.model_responses.rename({
                        "agent_type": "agentType"
                    }, axis=1)

            self.model_responses = self.model_responses.drop(['correct_answer'], axis=1)
            # TODO: check if all questions csv have text input types
            self.model_responses = self.model_responses.merge(self.questions[['image_file','question','correct_answer']])
            self.model_responses = self.model_responses.dropna(subset=["agent_response"])

            if print_stats:
                for model_config in model_configs:
                    agent_type = model_config['model']
                    agent_responses = self.model_responses[self.model_responses["agentType"] == agent_type]
                    print(f"Agent {agent_type} only responded to {len(agent_responses) / len(self.questions)} questions for test {self.test_name}")

            if self.units_of_measure == "a_in_b":
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
                    
                self.model_responses["a_in_b"] = self.model_responses.apply(
                    check_answer, axis=1
                )

            # self.model_responses.to_csv(f"./single_accuracy_plots/{self.test_name}_used_df.csv")
            all_responses.append(self.model_responses)

        self.model_responses = pd.concat(all_responses)

    

    def bootstrap_ci(
            self, 
            raw_data, 
            n_iterations=1000,
            statistic=np.mean,
            agent=None):
        data = raw_data.copy()
        data["question_image"] = data["question"] + " & " + data["image_file"]
        rng = Generator(PCG64())
        questions = list(data["question_image"].unique())
        n_size = len(questions)
        df = data.copy()

        def bootstrap_iteration(data, chosen_qs):
            # data = df.copy() #.sample(frac=1, replace=True, random_state=1)
            filter_df = data[data["question_image"].isin(chosen_qs)] # Filter based on chosen questions
            # data = data.sample(frac=1, replace=True, random_state=1)
            bs_mean = statistic(filter_df[self.units_of_measure]) 
            return (bs_mean, list(chosen_qs))

        qset_means = Parallel(n_jobs=-1)(
            delayed(bootstrap_iteration)(
                # df.sample(frac=1, replace=True, random_state=1),
                df.copy(),
                rng.choice(questions, n_size,  replace=True)
            ) for _ in range(n_iterations)
        )
        
        means = []
        qs_used = []
        means = [bs_mean for bs_mean, chosen_qs in qset_means]

        if agent:
            pd.DataFrame({"means": [qset_means], "qs_used": [qs_used], "agent": agent}).to_csv(f"./single_accuracy_plots/{self.test_name}_{agent.replace("/", "_").replace(".", "_")}.csv")
        
        # 95% confidence interval
        lower = np.percentile(means, 2.5)
        upper = np.percentile(means, 97.5)
        
        return lower, upper


    def create_confidence_interval_df(self, data, statistic=np.mean):
        data_list = []
        num_questions = len(self.questions[["question", "image_file"]])

        for agent in data["agentType"].unique():
            agent_res = data[data["agentType"] == agent]

            lower, upper = self.bootstrap_ci(agent_res, statistic=statistic, agent=agent)

            data_list.append({
                "agentType": agent,
                "ci_upper": upper, 
                "ci_lower": lower,
                "value_count": len(agent_res[['question', 'image_file']].value_counts()) / num_questions,
                "mean": statistic(agent_res[self.units_of_measure])
            })

        return pd.DataFrame(data_list)

    def create_accuracy_df(self, data, statistic):
        if statistic == np.mean:
            acc_df = data.groupby(["question", "image_file", "agentType"])[self.units_of_measure].mean()
        else:
            acc_df = data.groupby(["question", "image_file", "agentType"])[self.units_of_measure].median()
        return acc_df.reset_index()
    
    def create_error_scatter_plot(self):
        model_order = [c[0] for c in self.model_config]
        color_order = [c[1] for c in self.model_config]

        if self.units_of_measure == "absolute_error":
            self.model_responses = self.model_responses[self.model_responses['absolute_error'] <= 1000000]
            self.model_responses = self.model_responses[self.model_responses['absolute_error'] >= 0.001]
            scale=alt.Scale(
                domain=[0, 1000000],
                type="log"
            )
        elif self.units_of_measure == "minmax_axis_normalized_error":
            self.model_responses = self.model_responses[self.model_responses['minmax_axis_normalized_error'] <= 1]
            self.model_responses = self.model_responses[self.model_responses['minmax_axis_normalized_error'] >= 0]
            scale=alt.Scale()
            # domain = [0, 1]
        
        accuracy_df = self.create_accuracy_df(self.model_responses)
        scatter_plot = alt.Chart(accuracy_df, title=self.test_name).mark_circle(size=8, opacity=0.7).encode(
            y=alt.Y(f"{self.units_of_measure}:Q", title=f"{self.units_of_measure}", scale=scale), #, 
            x=alt.X("agentType:N", scale=alt.Scale(domain=model_order), title=None),
            xOffset="jitter:Q",
            color=alt.Color('agentType:N', scale=alt.Scale(domain=model_order, range=color_order)).legend(None),
            ).transform_calculate(
                jitter="sqrt(-2*log(random()))*cos(2*PI*random())" 
            )

        return scatter_plot

    def create_error_plot(self, hide_y_title=False, render_ci=True, max_point=False, statistic=np.mean):
        """ Creates a scatter plot with error bars for accuracy data.

        Args:
            test (str): name of the test
        """
        accuracy_df_items = self.create_accuracy_df(self.model_responses, statistic=statistic)
        all_questions = []
        for test_name in ['ggr','vlat','calvi-trick','calvi-standard']:
            questions = pd.read_csv(f'{self.AWS_PREFIX}/{test_name}/questions.csv')
            all_questions.append(questions)
        all_questions = pd.concat(all_questions)
        all_questions['chart_type'] = all_questions['chart_type'].apply(lambda r : r.lower())
        all_questions['chart_type'] = all_questions['chart_type'].replace({"100 % stacked bar chart": "100% stacked bar chart"})

        all_accuracy_df = pd.merge(all_questions, accuracy_df_items, on=['question', 'image_file'])
        

        ytitle = self.units_of_measure if not hide_y_title else None
        model_order = [c['model'] for c in self.model_configs]
        main_color_order = [c['color'] for c in self.model_configs]
        bg_color_order = [c['accent_color'] for c in self.model_configs]

        self.model_responses = pd.merge(all_questions, self.model_responses,  on=['question', 'image_file'])
        
        chart_types = [
            'bar chart',  'histogram', 'stacked bar chart', '100% stacked bar chart', 'pie chart',
            'line chart',  'area chart', 'stacked area chart', 'treemap', 'choropleth map',
            'dot plot', 'scatterplot',  'bubble chart',
       ]
        all_model_responses = self.model_responses.copy()

        all_plots = []
        for chart_type in chart_types:
            self.model_responses = all_model_responses[all_model_responses['chart_type'] == chart_type]
            accuracy_df = all_accuracy_df[all_accuracy_df['chart_type'] == chart_type]

            if render_ci:
                # try:
                #     ci_df = pd.read_csv(f'./ci_df/{self.test_name}.csv')
                # except:
                ci_df = self.create_confidence_interval_df(self.model_responses, statistic=statistic)
                ci_df.to_csv(f'./ci_df/{self.test_name}.csv')

                error_bars = alt.Chart(ci_df).mark_errorbar().encode(
                    x=alt.X("agentType:N", title=None),
                    y=alt.Y("ci_upper", title=None),
                    y2=alt.Y2("ci_lower"),
                    strokeWidth=alt.value(2),
                    color=alt.Color('agentType', scale=alt.Scale(domain=model_order, range=main_color_order)).legend(None)
                )

                mean_point_plot = alt.Chart(ci_df, title=self.test_name).mark_point(
                    size=80, 
                    # size=10,
                    filled=True,
                    strokeWidth=1,
                    opacity=1
                ).encode(
                    y=alt.Y("mean:Q", ),
                    x=alt.X("agentType:N", scale=alt.Scale(domain=model_order), title=None),
                    color=alt.Color(
                        'agentType:N', 
                        scale=alt.Scale(
                            domain=model_order, 
                            range=main_color_order
                        )
                    ).legend(None),
                )

                mean_bar_plot = alt.Chart(ci_df, title=self.test_name).mark_bar(
                    size=10,
                    filled=True,
                    strokeWidth=1,
                    opacity=1
                ).encode(
                    y=alt.Y("mean:Q", scale=alt.Scale(domain=[0,1])),
                    x=alt.X("agentType:N", scale=alt.Scale(domain=model_order), title=None),
                    color=alt.Color(
                        'agentType:N', 
                        scale=alt.Scale(
                            domain=model_order, 
                            range=main_color_order
                        )
                    ).legend(None),
                )

            if max_point and render_ci:
                max_point = alt.Chart(ci_df).mark_line().encode(
                    x=alt.X("agentType:N", scale=alt.Scale(domain=model_order)),
                    y=alt.Y("value_count"),
                    strokeWidth=alt.value(2.5),
                    shape=alt.Shape('agentType:N').legend(None),
                    color=alt.value("#2b2b2b") #alt.Color('agentType', scale=alt.Scale(domain=order, range=color_arr)).legend(None)
                )

            if self.units_of_measure == "a_in_b":
                y_domain = [0, 1]
            elif self.units_of_measure == "minmax_axis_normalized_error":
                max_point = ci_df["ci_upper"].max()
                max_point = 1.4 if max_point < 1.4 else max_point

                accuracy_df = accuracy_df[accuracy_df[self.units_of_measure] <= max_point]
                y_domain = [0, max_point]
            
            if self.units_of_measure == "minmax_axis_normalized_error":
                opacity = 0.5
            else:
                opacity = 1

            scatter_plot = alt.Chart(accuracy_df, title=self.test_name).mark_circle(size=16,opacity=opacity).encode(
                y=alt.Y(f"{self.units_of_measure}:Q", title=ytitle, scale=alt.Scale(domain=y_domain)),
                x=alt.X("agentType:N", scale=alt.Scale(domain=model_order), title=None),
                xOffset="jitter:Q",
                color=alt.Color('agentType:N', scale=alt.Scale(domain=model_order, range=bg_color_order)).legend(None),
            ).transform_calculate(
                jitter="sqrt(-2*log(random()))*cos(2*PI*random())" 
            )

            final_plot = scatter_plot
            
            if render_ci:
                final_plot = final_plot + error_bars + mean_point_plot
                # final_plot = mean_bar_plot
                graph_count = all_questions[all_questions['chart_type'] == chart_type].shape[0]
                final_plot = final_plot.properties(title=f"{chart_type} ({graph_count} items)", height=440.8312, width=183.6797).resolve_scale(color="independent",  opacity="independent")
                _dir = f'./single_accuracy_plots/{self.prompt_type}/graphtype-acc'
                if not os.path.exists(_dir):
                    os.makedirs(_dir)

                file_name = f"{_dir}/accuracy_plot_med_p00-point-{chart_type.replace(" ", "-")}.pdf"
                final_plot.save(file_name)

                print(f"Saving plot to {file_name}")

            all_plots.append(final_plot)
        
        alt.hconcat(*all_plots).save(f"{_dir}/1accuracy_plot_med_p00-point-all_chart_types.pdf")

        return alt.hconcat(*all_plots)

def create_single_model_comparison_plots(contrast_dict=None):

    model_configs = [
        {'model': "Salesforce/blip2-flan-t5-xl", 'top_p': 'p06', 'temp': 't1', 'color': '#5ba3cf', 'accent_color': '#8cbedd'},
        {'model': "Salesforce/blip2-flan-t5-xxl", 'top_p': 'p1', 'temp': 't10', 'color': '#4c78a8', 'accent_color': '#81a0c2'},
        {'model': "llava-hf/llava-1.5-7b-hf", 'top_p': 'p04', 'temp': 't1', 'color': '#f9b574', 'accent_color': '#facb9d'},
        {'model': "llava-hf/llava-1.5-13b-hf", 'top_p': 'p1', 'temp': 't04', 'color': '#f58518', 'accent_color': '#f79d46'},
        {'model': "liuhaotian/llava-v1.6-34b", 'top_p': 'p1', 'temp': 't04', 'color': '#BF9000', 'accent_color': '#F1C232'},
        {'model': "google/pix2struct-chartqa-base", 'top_p': 'p08', 'temp': 't1', 'color': '#b9a7d0', 'accent_color': '#c7b8d9'},
        {'model': "google/matcha-chartqa", 'top_p': 'p04', 'temp': 't1', 'color': '#8b6db2', 'accent_color': '#a28ac1'},
        {'model': "GPT-4V", 'top_p': 'p1', 'temp': 't02', 'color': '#b85536', 'accent_color': '#C6765E'},
        {'model': "Human", 'top_p': 'pna', 'temp': 'tna', 'color': '#2e693b', 'accent_color': '#73D287'},
    ]
    prompt_type = "indist_instructions_question"

    accuracy_plot = AccuracyPlot(
        [
            "ggr",
            "vlat",
            'calvi-trick',
        ], 
        prompt_type,
        model_configs=model_configs,
        units_of_measure='a_in_b',
        print_stats=True,
    )
                
    plot = accuracy_plot.create_error_plot(
        render_ci=True,
        statistic=np.mean
    ).properties(title=f"graphtype-acc") #, height=300, width=125
    _dir = f'./single_accuracy_plots/{prompt_type}/graphtype-acc'
    if not os.path.exists(_dir):
        os.makedirs(_dir)

    file_name = f"{_dir}/accuracy_plot_med_p00-3.pdf"
    plot.save(file_name)

    print(f"Saving plot to {file_name}")

if __name__ == "__main__":
    create_single_model_comparison_plots()
