import re
# from sklearn.metrics.pairwise import cosine_similarity
# 
import numpy as np

try:
    from sentence_transformers import SentenceTransformer, util
    sentence_embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device="cpu")
except:
    print("Could not import SentenceTransformer. Please install the library using 'pip install sentence-transformers'")

class EvaluationMetrics:

    def clean_word(self, word):
        """Clean the text by removing special characters and converting to lowercase."""
        if (word == None) or (word == np.nan) or (word == ""):
            return np.nan

        cleaned_word = str(word).lower()
        cleaned_word = cleaned_word.replace("\n", "")
        cleaned_word = cleaned_word.strip()
        cleaned_word = cleaned_word.replace(",", "")
        cleaned_word = cleaned_word.replace("%", "") # remove percentage signs, i.e 25% -> 25

        return cleaned_word
    
    def remove_prompt(self, text, prompt):
        """Remove the prompt from the text, including special tokens."""
        if text == "" or text == None or text == np.nan or text == "nan":
            return np.nan
        
        # attempt 1: remove raw prompt
        response = str(text).replace(prompt, "")

        # inital prompt cleaning to remove prompt
        formatted_prompt = str(prompt)
        formatted_prompt = formatted_prompt.replace("<image>", "").replace("\n", "").replace(",", "")
        formatted_prompt = formatted_prompt.lower()
        response = response.replace(formatted_prompt, "")
        
        # additional cleaning to remove prompt
        formatted_prompt = formatted_prompt.replace("user: ", "user:  ")
        response = response.replace(formatted_prompt, "")
        
        response = response.replace("<image>", "")

        if response == "":
            response = np.nan

        return response

    def extract_numerical_responses(self, word):
        """Extract all numbers from the response."""
        # remove all commas for example 1,000 -> 1000
        num_answer = str(word).replace(",", "")

        # find all negative numbers and decimals 
        pattern = r'-?\d+(\.\d+)?'
        all_numbers = list(re.finditer(pattern, num_answer))
        all_numbers = [float(num.group()) for num in all_numbers if num != '']

        return all_numbers

    def jaccard_similarity(self, word, word2):
        """Calculate the Jaccard Similarity (intersection over union) between two sets of words."""
        set1 = set([self.clean_word(word) for word in word.split()])
        set2 = set([self.clean_word(word) for word in word2.split()])
        
        intersection = set1.intersection(set2) # intersect the set
        union = set1.union(set2) 
        return len(intersection) / len(union)
    
    def a_in_b(self, word, word2):
        """Check if a word contains another word."""
        cleaned_word = self.clean_word(str(word))
        cleaned_word2 = self.clean_word(str(word2))

        return cleaned_word in cleaned_word2 #int(str(word) in str(word2))
    
    def relaxed_accuracy(self, word, correct_answer, e=0.05):
        """Check if the response is within a certain error rate of the correct answer."""

        numerical_answers = self.extract_numerical_responses(word)
        correct_numerical_answer= float(correct_answer) 
        is_correct = 0
        for numerical_answer in numerical_answers:
            if abs(numerical_answer - correct_numerical_answer) <= (e*abs(correct_numerical_answer)):
                is_correct = 1

        return is_correct

    def get_relative_error(self, response, correct_answer):
        """Calculate the absolute relative error between the response and the correct answer."""
        
        # scan through all numbers in string a pick the one with the lowest error
        numerical_answers = self.extract_numerical_responses(response)
        correct_numerical_answer = float(correct_answer)
        errors = [abs(numerical_answer - correct_numerical_answer) for numerical_answer in numerical_answers]
        if len(errors) == 0: # if no numbers are found in the response
            return np.nan
        response_error_idx = np.argmin(errors)
        response_error = correct_numerical_answer - errors[response_error_idx] 

        return response_error
    
    def get_best_numerical_response(self, response, correct_answer):
        """
        Selects the best numerical response from the response based on the absolute relative error.
        """
        numerical_answers = self.extract_numerical_responses(response)
        correct_numerical_answer = float(correct_answer)
        errors = [abs(numerical_answer - correct_numerical_answer) for numerical_answer in numerical_answers]
        if len(errors) == 0: # if no numbers are found in the response
            return np.nan
        
        min_error_idx = np.argmin(errors)
        return numerical_answers[min_error_idx]
    
    def get_absolute_error(self, response, correct_answer):
        """Calculate the absolute relative error between the response and the correct answer."""
        
        # scan through all numbers in string a pick the one with the lowest error
        numerical_answers = self.extract_numerical_responses(response)
        correct_numerical_answer = float(correct_answer)
        errors = [abs(numerical_answer - correct_numerical_answer) for numerical_answer in numerical_answers]
        if len(errors) == 0: # if no numbers are found in the response
            return np.nan
        response_error = min(errors)
        
        return response_error
    
    def normalize_by_correct_answer(self, response, correct_answer):
        """Normalize the error by the correct answer."""
        correct_answer = float(correct_answer)
        if correct_answer == 0:
            return np.nan
        else:
            return self.get_absolute_error(response, correct_answer) / float(correct_answer)
        
    def minmax_normalized_error(self, response_error, group_min, group_max):
        """Calculate the error rate between the response and the correct answer."""
        return (response_error - group_min) / (group_max - group_min)

    def one_minus_minmax_normalized_error(self, response_error, group_min, group_max):
        """Calculate the error rate between the response and the correct answer."""

        return 1 - ((response_error - group_min) / (group_max - group_min))
    
    def one_minus_zscore_normalization_error(self, response_error, group_mean, group_std):
        """Calculate the z-score normalized error between the response and the correct answer."""
        
        return 1 - ((response_error - group_mean) / group_std)
    
    def sbert_embedding(self, response):
        return sentence_embedding_model.encode(sentences=[response])

    # def sbert_embedding_cosine_similarity(self, response1, response2):
    #     """Calculate the cosine similarity between two responses using SBERT."""
    #     embedding1, embedding2 = sentence_embedding_model.encode(sentences=[response1, response2])

    #     return cosine_similarity(embedding1, embedding2)
        

    def cosine_similarity(self, embedding1, embedding2):
        """Calculate the cosine similarity between two sets of words."""

        return util.cos_sim(embedding1, embedding2)
        
        # dot_product = np.dot(embedding1, embedding2)

        # norm_1 = np.linalg.norm(embedding1)
        # norm_2 = np.linalg.norm(embedding2)

        # cosine_similarity = dot_product / (norm_1 * norm_2)

        # return cosine_similarity


if __name__ == "__main__":
    eval_metrics = EvaluationMetrics()
    word = "The answer is 1,000"
    word2 = "The answer is 1000"
    print(eval_metrics.jaccard_similarity(word, word2))