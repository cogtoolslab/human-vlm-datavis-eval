import unittest
from unittest.mock import patch, MagicMock
from PIL import Image
import pandas as pd
import requests
from data_utils import DataUtils

class TestDataUtils(unittest.TestCase):
    def test_fetch_instructions_success(self):
        test_type = "example"
        expected_url = f"https://data-visualization-benchmark.s3.us-west-2.amazonaws.com/{test_type}/instructions.txt"
        expected_text = "Sample instructions"
        with patch("requests.get") as mock_get:
            mock_get.return_value.text = expected_text
            result = DataUtils.fetch_instructions(test_type)
            mock_get.assert_called_once_with(expected_url)
            self.assertEqual(result, expected_text)

    def test_fetch_instructions_error(self):
        test_type = "example"
        expected_url = f"https://data-visualization-benchmark.s3.us-west-2.amazonaws.com/{test_type}/instructions.txt"
        expected_error = "Error fetching instructions"
        with patch("requests.get") as mock_get:
            mock_get.side_effect = requests.RequestException(expected_error)
            result = DataUtils.fetch_instructions(test_type)
            mock_get.assert_called_once_with(expected_url)
            self.assertIsNone(result)

    def test_fetch_questions_success(self):
        test_type = "example"
        expected_url = f"https://data-visualization-benchmark.s3.us-west-2.amazonaws.com/{test_type}/questions.csv"
        expected_data = pd.DataFrame({"question": ["Q1", "Q2"], "answer": ["A1", "A2"]})
        with patch("pd.read_csv") as mock_read_csv:
            mock_read_csv.return_value = expected_data
            result = DataUtils.fetch_questions(test_type)
            mock_read_csv.assert_called_once_with(expected_url)
            pd.testing.assert_frame_equal(result, expected_data)

    def test_fetch_questions_error(self):
        test_type = "example"
        expected_url = f"https://data-visualization-benchmark.s3.us-west-2.amazonaws.com/{test_type}/questions.csv"
        expected_error = "Error loading questions"
        with patch("pd.read_csv") as mock_read_csv:
            mock_read_csv.side_effect = Exception(expected_error)
            result = DataUtils.fetch_questions(test_type)
            mock_read_csv.assert_called_once_with(expected_url)
            self.assertIsInstance(result, pd.DataFrame)
            self.assertTrue(result.empty)

    def test_post_data_success(self):
        expected_url = 'https://foraes.com/api/v1/prompt'
        expected_data = {"key": "value"}
        expected_response = MagicMock()
        with patch("requests.post") as mock_post:
            mock_post.return_value = expected_response
            result = DataUtils.post_data(expected_data)
            mock_post.assert_called_once_with(expected_url, json=expected_data)
            self.assertEqual(result, expected_response)

    def test_post_data_error(self):
        expected_url = 'https://foraes.com/api/v1/prompt'
        expected_data = {"key": "value"}
        expected_error = "Error posting data"
        with patch("requests.post") as mock_post:
            mock_post.side_effect = requests.RequestException(expected_error)
            result = DataUtils.post_data(expected_data)
            mock_post.assert_called_once_with(expected_url, json=expected_data)
            self.assertIsNone(result)

    def test_get_image_success(self):
        expected_url = 'https://example.com/image.jpg'
        expected_image = MagicMock(spec=Image.Image)
        with patch("requests.get") as mock_get, patch("Image.open") as mock_open:
            mock_get.return_value.raw = MagicMock()
            mock_open.return_value.convert.return_value = expected_image
            result = DataUtils.get_image(expected_url)
            mock_get.assert_called_once_with(expected_url, stream=True)
            mock_open.assert_called_once_with(mock_get.return_value.raw)
            self.assertEqual(result, expected_image)

    def test_get_image_error(self):
        expected_url = 'https://example.com/image.jpg'
        expected_error = "Error getting image"
        with patch("requests.get") as mock_get, patch("Image.open") as mock_open:
            mock_get.side_effect = requests.RequestException(expected_error)
            result = DataUtils.get_image(expected_url)
            mock_get.assert_called_once_with(expected_url, stream=True)
            self.assertIsNone(result)

if __name__ == "__main__":
    unittest.main()