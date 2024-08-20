import numpy as np
from sentence_transformers import util
import pandas as pd
from utils import prepare_unprocessed_response_dataframe, read_model_configs
from evaluation_metrics import EvaluationMetrics
from sentence_transformers import SentenceTransformer, util
sentence_embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device="cpu")

def get_sbet_embedding(responses):
    return np.asarray(sentence_embedding_model.encode(sentences=responses))

def get_embedding_df():
    try:
        all_model_responses = pd.read_csv("./raw_response_embeddings/raw_responses.csv")
    except:
        tests = ['ggr', 'vlat', 'calvi-trick', 'holf', 'holf2', 'chartqa-test-continuous-human']
        # tests = ['vlat']
        all_model_responses = []
        for test in tests:
            model_responses = prepare_unprocessed_response_dataframe(test) 
            model_responses['sbert_embedding'] = model_responses['agent_response'].apply(get_sbet_embedding)
            all_model_responses.append(model_responses)
            model_responses.to_csv(f"./raw_response_embeddings/raw_responses_{test}.csv")
            print(test)

        all_model_responses = pd.concat(all_model_responses).reset_index()
        all_model_responses.to_csv("./raw_response_embeddings/raw_responses.csv")

    all_model_responses['agent_type'] = all_model_responses['agentType']
    all_model_responses['item'] = all_model_responses['question'] + " & " + all_model_responses['image_file']

    return all_model_responses

def within_between_question_matrix(model_responses, agent_1, agent_2):
    agent_1_res = model_responses[model_responses["agent_type"] == agent_1]
    agent_2_res = model_responses[model_responses["agent_type"] == agent_2]
    
    items = list(model_responses[["item"]].drop_duplicates()['item'])
    for item in items:
        agent_1_item = agent_1_res[agent_1_res['item'] == item]
        agent_1_embeddings = np.vstack(agent_1_item['sbert_embedding'])

        agent_2_items = agent_2_res[agent_2_res['item'] != item]
        agent_2_embeddings = np.vstack(agent_2_items['sbert_embedding'])

        agent_12_items = agent_2_res[agent_2_res['item'] == item]
        agent_12_embeddings = np.vstack(agent_12_items['sbert_embedding'])

        cos_sim_matrix = util.cos_sim(agent_1_embeddings, agent_2_embeddings)
        
    # embeddings = np.vstack(model_responses['raw_response_sbert_embedding'])
    
    # item_iteration_count = 10
    
    test_item_count = len(items)
    print(cos_sim_matrix.max(),cos_sim_matrix.min(), agent_1, agent_2)

    current_item = 0
    item_means = []
    while current_item < test_item_count:
        item_iteration_count = items.iloc[current_item]['count']
        
        # Get the block of the matrix with iterations on the same question
        block_start = current_item*item_iteration_count
        block_end = (current_item+1)*item_iteration_count
        within_question_matrix = cos_sim_matrix[block_start:block_end,block_start:block_end]

        # get upper triangle of the matrix
        within_question_unique = np.triu(within_question_matrix, k=1)
        # number of values in upper triangle
        triu_count = (item_iteration_count * (item_iteration_count - 1)) / 2
        
        within_question_mean = np.sum(within_question_unique) / triu_count

        # Get the block of the upper triangle of the matrix with iterations on different questions
        between_questions_before_matrix = cos_sim_matrix[block_start:block_end, 0:block_start]
        between_questions_after_matrix = cos_sim_matrix[block_start:block_end, block_end:]
        between_question_matrix = np.concatenate((between_questions_before_matrix, between_questions_after_matrix), axis=1)

        # get mean between questions
        between_question_mean = np.mean(between_question_matrix.flatten())

        item_means.append(within_question_mean - between_question_mean)
        current_item += 1
    
    return item_means

if __name__ == '__main__':
    model_responses = get_embedding_df()
    
    model_responses['sbert_embedding'] = model_responses['sbert_embedding'].apply(lambda e : np.fromstring(e))
    models = [mc['model'] for mc in read_model_configs()]
    rows = []
    for m1 in models:
        for m2 in models:
            if m1 == "Human" or m2 == "Human":
                continue
            # if m1 == m2:
            #     continue
            agent_item_means = within_between_question_matrix(
                model_responses=model_responses,
                agent_1=m1,
                agent_2=m2
            )
            print(m1,m2)
            rows.append([m1,m2,agent_item_means, np.mean(agent_item_means)])

    pd.DataFrame(rows, 
        columns=['agent1', 'agent2', 'item_means', 'pair_mean']
    ).to_csv("./raw_response_embeddings/within_between_sim.csv")
    