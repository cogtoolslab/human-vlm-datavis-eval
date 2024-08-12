import pandas as pd
import numpy as np
import altair as alt
from evaluation_metrics import EvaluationMetrics
from aws_upload import upload_file_to_s3
import os

AWS_PREFIX = "https://data-visualization-benchmark.s3.us-west-2.amazonaws.com"

def process_extracted_responses(model_sets):
    models = [model[0] for model in model_sets]
    prompt_type = 'indist_instructions_question'
    top_p_dir = 'p04'
    temperature_dir = 't1'

    test_types = [
        "ggr",
        "vlat",
        "holf"
    ]
    

    model_responses = []
    for model in models:
        _dir = f'responses/{prompt_type}/{top_p_dir}/{temperature_dir}/{model.replace("/", "-")}'
        for test_type in test_types:
            file_name = f'https://data-visualization-benchmark.s3.us-west-2.amazonaws.com/{test_type}/{_dir}/extracted_responses.csv'
            print(f"{file_name}")
            model_responses.append(pd.read_csv(file_name))

    model_responses = pd.concat(model_responses)

    questions = []
    for test_type in test_types:
        file_name = f"https://data-visualization-benchmark.s3.us-west-2.amazonaws.com/{test_type}/questions.csv"
        print(file_name)
        questions.append(pd.read_csv(file_name))

    questions = pd.concat(questions)

    def has_incorrect_response(row):
        r = str(row["agent_response"])
        answer = str(row["correct_answer"]).lower()
        response = str(row["agent_response"]).lower()
        test_type = row["testType"]

        if test_type == "holf":
            return EvaluationMetrics().get_best_numerical_response(response, answer)

        check1 = (
            r.find("The model response does not provide") != -1 or
            r.find("response does not") != -1 or
            r.find("The extracted answer") != -1 or
            r.find("The model") != -1 or
            response.find("error") != -1 or
            response.find("sorry") != -1 or
            response.find("does not provide") != -1 or
            r.find("To provide an accurate answer") != -1 or 
            r.find("does not include") != -1 or
            r.find("I need the actual percentages") != -1 or
            r.find("Unable to extract") != -1 or
            r.find("model response did not") != -1 or
            r.find("model response itself is not provided") != -1
        )

        if response.find(answer) != -1:
            return answer
        
        if check1:
            return np.nan

        item = questions[
            (questions["question"] == row["question"]) &
            (questions["image_file"] == row["image_file"])                
        ].iloc[0]
        textInput = item["text_input_type"]

        if textInput == "multiple_choice_question":
            mc1, mc2, mc3, mc4 = item["mc1"], item["mc2"], item["mc3"], item["mc4"]

            for mc in [mc4,mc3,mc2,mc1]:
                if mc != np.nan and response.find(str(mc).lower().strip()) != -1:
                    return mc
        
            return np.nan
        elif textInput == "open_ended_question" and test_type == "ggr":
            return EvaluationMetrics().get_best_numerical_response(response, answer)

    model_responses["agent_response"] = model_responses.apply(has_incorrect_response, axis=1)

    # Code to upload to lab S3 server -- commented out for cogsci release
    # for test_type in test_types:
        # _dir = f'responses/{prompt_type}/{top_p_dir}/{temperature_dir}'
        # file_name = f'{_dir}/processed_extracted_responses.csv'
        # model_responses.to_csv(file_name, index=False)
        # upload_file_to_s3(file_name, test_type, f'{_dir}/processed_extracted_responses.csv')

    charts = []
    for i, model in enumerate(model_sets):
        agent_responses =  model_responses[model_responses["agentType"] == model[0]].copy()[["testType", "agent_response"]].reset_index()
        agent_responses["is_valid"] = agent_responses["agent_response"].notna().astype(int)
        agent_responses = agent_responses.groupby(["testType"])["is_valid"].aggregate(lambda g : sum(g) / len(g))

        agent_responses = agent_responses.reset_index()

        y_axis = None if i != 0 else alt.Axis()

        # _dir = f"../results/dataframe/valid_bar_df/{model[0].replace("/", "_")}"
        _dir = f"../results/dataframe/valid_bar_df"
        if not os.path.exists(_dir):
            os.makedirs(_dir)
        # agent_responses.to_csv(f"{_dir}/{test_type}.csv")
        agent_responses.to_csv(f"{_dir}/{model[0].replace("/", "_")}.csv")

        chart = alt.Chart(agent_responses).mark_bar(color=model[1]).encode(
            x=alt.X('testType', title=None, scale=alt.Scale(domain=["ggr", "vlat", "holf"])),
            y=alt.Y('is_valid', title=None, axis=y_axis, scale=alt.Scale(domain=[0, 1]) ),
            color=alt.Color(legend=None),
        ).properties(width=40, height=100, title=alt.Title(
            "",
            anchor="middle"
        ))
        charts.append(chart)

    _dir = "../results/figures/valid_bar_plots"
    file_name = f'{_dir}/processed_responses_valid.pdf'
    alt.hconcat(*charts, spacing=10).save(file_name)
    print(file_name)



if __name__ == "__main__":
    model_sets = [
        ('Salesforce/blip2-opt-2.7b', '#969696', '#5ba3cf'),
        ('llava-hf/llava-1.5-7b-hf', '#f58518', '#f9b574'),
        ('Salesforce/blip2-flan-t5-xl', '#5ba3cf', '#5ba3cf'),
        ('Salesforce/blip2-flan-t5-xxl', '#4c78a8', '#4c78a8'),
        ('GPT-4V', '#b85536', '#b85536'),
    ]
    process_extracted_responses(model_sets=model_sets)