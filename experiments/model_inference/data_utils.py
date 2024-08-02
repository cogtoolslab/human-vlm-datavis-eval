import requests
import pandas as pd
from PIL import Image

class DataUtils:
    @staticmethod
    def fetch_instructions(testType):
        try:
            url = f"https://data-visualization-benchmark.s3.us-west-2.amazonaws.com/{testType}/instructions.txt"
            return requests.get(url).text
        except requests.RequestException as e:
            print(f"Error fetching {url}: {e}")
            return None

    @staticmethod
    def fetch_questions(testType):
        try:
            url = f"https://data-visualization-benchmark.s3.us-west-2.amazonaws.com/{testType}/questions.csv"
            return pd.read_csv(url)
        except Exception as e:
            print(f"Error loading questions from {url}: {e}")
            return pd.DataFrame()

    @staticmethod
    def post_data(data):
        try:
            url = 'https://foraes.com/api/v1/prompt'
            response = requests.post(url, json=data)
            return response
        except requests.RequestException as e:
            return None

    @staticmethod
    def get_image(url):
        try:
            return Image.open(requests.get(url, stream=True).raw).convert('RGB')
        except requests.RequestException as e:
            return None
