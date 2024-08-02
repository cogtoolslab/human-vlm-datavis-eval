from .benchmark import Benchmark
import time
import os


print("PID: ", os.getpid())

prompt_types = [
    "instructions_question", 
    #"indist_instructions_question",
    #"indist_instructions_cot_0shot_question"
    # "incontext_instructions_question"
]
datasets=[
    # 'chartqa-test-continuous',
    #"ggr",
    #"vlat",
    #'calvi-trick',
    #'calvi-standard',
    #'holf2',
    "holf",
    #'chartqa-test-continuous',
    #'chartqa-test-categorical',
]

models = [
    "GPT-4V",
    #"Salesforce/blip2-flan-t5-xxl", 
    #"llava-hf/llava-1.5-13b-hf",
    #"liuhaotian/llava-v1.6-34b",
    # "google/matcha-chartqa",

    #"Salesforce/blip2-flan-t5-xl",
    #"llava-hf/llava-1.5-7b-hf", 
    #"liuhaotian/llava-v1.6-vicuna-13b",
    # "liuhaotian/llava-v1.6-mistral-7b",
    #"Salesforce/blip2-opt-2.7b",
    #"google/pix2struct-chartqa-base",
    # "google/deplot"
]

# num_trials = 10
#num_trials = 10
num_trials = 2
#num_trials = 8
#temperatures = [0.2, 0.4, 0.6, 0.8, 1][::-1]
#temperatures = [0.2, 0.6, 0.8][::-1]
temperatures = [1]
top_ps = [0.4]
#top_ps = [0.2, 0.4, 0.6, 0.8][::-1]
#top_ps = [0.4, 0.8]
# top_ps = [0.2, 0.6][::-1]
#top_ps = [0.2, 0.4]

for model in models:
    for dataset in datasets:
        for prompt_type in prompt_types:
            timeout = 0
            if model == "GPT-4V":
                timeout = 7
            print(f"Running {model} on {dataset} with {prompt_type} prompts")
            # for t in temperatures:
            #     benchmark = Benchmark(model, dataset, prompt_type, numTrials=num_trials, timeout=timeout, temperature=t, topP=1)
            #     benchmark.run()
            for p in top_ps:
                benchmark = Benchmark(model, dataset, prompt_type, numTrials=num_trials, timeout=timeout, temperature=1, topP=p)
                benchmark.run()
