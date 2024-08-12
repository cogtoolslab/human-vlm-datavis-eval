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
        testName, 
        prompt_type, 
        units_of_measure = 'a_in_b', 
        top_p_dir='p04', 
        temperature_dir='t1', 
        model_config=[],
        print_stats=False,
        relax_value = 0.10
    ):
        self.test_name = testName
        self.AWS_PREFIX = "https://data-visualization-benchmark.s3.us-west-2.amazonaws.com"
        self.questions = pd.read_csv(f'{self.AWS_PREFIX}/{testName}/questions.csv')
        self.top_p_dir = top_p_dir
        self.temperature_dir = temperature_dir
        self.evaluation_metric = EvaluationMetrics()
        self.units_of_measure = units_of_measure
        _dir = f'responses/{prompt_type}/{top_p_dir}/{temperature_dir}'
        raw_model_response_url =  f'{self.AWS_PREFIX}/{testName}/responses/{prompt_type}/{top_p_dir}/{temperature_dir}/model_responses.csv'
        model_response_url = f'https://data-visualization-benchmark.s3.us-west-2.amazonaws.com/{testName}/{_dir}/processed_extracted_responses.csv'
        
        print(model_response_url)
        self.model_responses = pd.read_csv(model_response_url)
        self.model_responses = self.model_responses[self.model_responses["testType"] == self.test_name]
        self.model_responses = self.model_responses.rename({
            "imageFile": "image_file"
        }, axis=1).dropna(subset=["agent_response"])

        self.human_responses = pd.read_csv(raw_model_response_url)
        self.human_responses = self.human_responses[(self.human_responses["agentType"] == "Human/Math-2-1") | 
                                                    (self.human_responses["agentType"] == "Human/Math-3") | 
                                                    (self.human_responses["agentType"] == "Human")]

        self.model_responses = pd.concat([self.model_responses, self.human_responses])
        self.model_responses["correct_answer"] = self.model_responses["correctAnswer"]
        
        self.is_numerical_test = self.test_name in ["chartqa-test-continuous", "holf", "holf2"]

        self.model_responses["agent_response"] = self.model_responses.apply(
            lambda r : self.evaluation_metric.remove_prompt(r["agent_response"], r["prompt"]),
            axis=1
        )

        if print_stats:
            for agent in model_config:
                agent_type = agent[0]
                agent_responses = self.model_responses[self.model_responses["agentType"] == agent_type]
                print(f"Agent {agent_type} only responded to {len(agent_responses) / len(self.questions)} questions for test {self.test_name}")

        if self.units_of_measure == "a_in_b":

            self.model_responses["a_in_b"] = self.model_responses.apply(
                lambda r: self.evaluation_metric.a_in_b(r['correct_answer'], r['agent_response']), axis=1
            )

        if self.units_of_measure == "minmax_axis_normalized_error":

            self.model_responses["agent_response"] = self.model_responses["agent_response"].astype(float)

            if relax_value > 0:
                top_quantile = self.model_responses["agent_response"].quantile(1-relax_value)
                bottom_quantile = self.model_responses['agent_response'].quantile(relax_value)

                def filter_outliers(row):
                    return (row['agent_response'] <= top_quantile) and (row['agent_response'] >= bottom_quantile)

                self.model_responses = self.model_responses[self.model_responses.apply(filter_outliers, axis=1)]

            self.model_responses["error"] = self.model_responses.apply(
                lambda r: self.evaluation_metric.get_absolute_error(r['agent_response'], r['correct_answer']), axis=1
            )
            self.model_responses = self.model_responses.dropna(subset=["error"])

            merged_response = self.questions[["question", "image_file", "min_label", "max_label"]]
            self.model_responses = self.model_responses.merge(merged_response, on=["question", "image_file"])
            
            self.model_responses[self.units_of_measure] = self.model_responses.apply(
                lambda r: self.evaluation_metric.minmax_normalized_error(r['error'], r['min_label'], r['max_label']), axis=1
            )

            self.model_responses = self.model_responses.dropna(subset=["minmax_axis_normalized_error"])
        
        self.model_config = model_config

        self.human_responses = None
        

    def bootstrap_ci(
            self, 
            raw_data, 
            n_iterations=10000,
            statistic=np.mean):
        data = raw_data.copy()
        data["question_image"] = data["question"] + " & " + data["image_file"]
        rng = Generator(PCG64())
        questions = list(data["question_image"].unique())
        n_size = len(questions)
        df = data.copy()

        # sample within the data
        df = df.sample(frac=1, replace=True, random_state=1)

        def bootstrap_iteration(data, chosen_qs):
            filter_df = data[data["question_image"].isin(chosen_qs)] # Filter based on chosen questions
            bs_mean = statistic(filter_df[self.units_of_measure]) #.mean() # Calculate mean of the filtered data
            return bs_mean
        means = Parallel(n_jobs=-1)(
            delayed(bootstrap_iteration)(df, rng.choice(questions, n_size,  replace=True)) for _ in range(n_iterations)
        )
        
        # 95% confidence interval
        lower = np.percentile(means, 2.5)
        upper = np.percentile(means, 97.5)
        
        return lower, upper


    def create_confidence_interval_df(self, data, statistic=np.mean):
        data_list = []
        num_questions = len(self.questions[["question", "image_file"]])

        for agent in data["agentType"].unique():
            agent_res = data[data["agentType"] == agent]

            lower, upper = self.bootstrap_ci(agent_res, statistic=statistic)

            data_list.append({
                "agentType": agent,
                "ci_upper": upper, 
                "ci_lower": lower,
                "value_count": len(agent_res[['question', 'image_file']].value_counts()) / num_questions,
                "mean": statistic(agent_res[self.units_of_measure])
            })

        return pd.DataFrame(data_list)

    def create_accuracy_df(self, data):
        acc_df = data.groupby(["question", "image_file", "agentType"])[self.units_of_measure].mean()
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
        # response_type = unit_of_measurement #if unit_of_measurement == "Proportion Correct" else "err"
        accuracy_df = self.create_accuracy_df(self.model_responses)

        ytitle = self.units_of_measure if not hide_y_title else None
        model_order = [c[0] for c in self.model_config]
        main_color_order = [c[1] for c in self.model_config]
        bg_color_order = [c[2] for c in self.model_config]
        

        if render_ci:
            try:
                ci_df = pd.read_csv(f'./ci_df/{self.test_name}.csv')
            except:
                ci_df = self.create_confidence_interval_df(self.model_responses, statistic=statistic)
                _dir = "../results/dataframe/accuracy_ci_df"
                if not os.path.exists(_dir):
                    os.makedirs(_dir)
                ci_df.to_csv(f'{_dir}/{self.test_name}.csv')

            error_bars = alt.Chart(ci_df).mark_errorbar().encode(
                x=alt.X("agentType:N"),
                y=alt.Y("ci_upper", title=ytitle),
                y2=alt.Y2("ci_lower"),
                strokeWidth=alt.value(2),
                color=alt.Color('agentType', scale=alt.Scale(domain=model_order, range=main_color_order)).legend(None)
            )

            mean_point_plot = alt.Chart(ci_df, title=self.test_name).mark_point(
                size=40, 
                filled=True,
                strokeWidth=1,
                opacity=1
            ).encode(
                y=alt.Y("mean:Q",),
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
            max_point = 1.2 if max_point < 1.2 else max_point

            accuracy_df = accuracy_df[accuracy_df[self.units_of_measure] <= max_point]
            y_domain = [0, max_point]
            # y_domain = [0, accuracy_df[self.units_of_measure].max()+1]
        
        _dir = "../results/dataframe/accuracy_scatter_df"
        if not os.path.exists(_dir):
            os.makedirs(_dir)
        accuracy_df.to_csv(f'{_dir}/{self.test_name}.csv')
        
        scatter_plot = alt.Chart(accuracy_df, title=self.test_name).mark_circle(size=8,).encode(
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
        
        return final_plot.resolve_scale(color="independent",  opacity="independent")


def create_single_model_comparison_plots(prompt_type="indist_instructions_question", temperature_dir='t1', top_p_dir='p04'):
    test_types = [
        "ggr",
        "vlat", 
        "holf",
    ]
    model_sets = [
        ('llava-hf/llava-1.5-7b-hf', '#f58518', '#f9b574'),
        ('Salesforce/blip2-flan-t5-xl', '#5ba3cf', '#8CBEDD'),
        ('Salesforce/blip2-flan-t5-xxl', '#4c78a8', '#6F93B9'),
        ('GPT-4V', '#b85536', '#CD8872'),
        ('Human/Math-2-1', "#639460", "#C8EBC6"),
        ('Human/Math-3', "#2e693b", "#73D287"),
    ]
    test_plots = []
    for test_type in test_types:
        relax_value = 0.00

        accuracy_plot = AccuracyPlot(
            test_type, 
            prompt_type, 
            top_p_dir=top_p_dir, 
            temperature_dir=temperature_dir,
            model_config=model_sets,
            units_of_measure='minmax_axis_normalized_error' if test_type in ["holf", "holf2"] else 'a_in_b',
            print_stats=True,
            relax_value=relax_value
        )
                    
        plot = accuracy_plot.create_error_plot(
            render_ci=True,
            statistic=np.median if test_type in ["holf", "holf2"] else np.mean
        ).properties(title=f"{test_type}", height=300, width=125)
        # _dir = f'./single_accuracy_plots/{prompt_type}/{test_type}/{top_p_dir}/{temperature_dir}'
        _dir = f'../results/figures/accuracy_plots'
        if not os.path.exists(_dir):
            os.makedirs(_dir)

        # file_name = f"{_dir}/accuracy_plot_med_p00-3.pdf"
        file_name = f"{_dir}/accuracy_plot_{prompt_type}-{test_type}-{top_p_dir}-{temperature_dir}.pdf"

        plot.save(file_name)
        # test_plots.append(plot)

        print(f"Saving plot to {file_name}")
    # test_plots = alt.vconcat(*test_plots).resolve_scale(
    #     x='shared'
    # ).configure_axis(
    #     labels=False
    # )

if __name__ == "__main__":

    create_single_model_comparison_plots()
