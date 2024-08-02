import unittest
from unittest.mock import patch
from benchmark import Benchmark

class TestBenchmark(unittest.TestCase):
    def setUp(self):
        self.benchmark = Benchmark()
        self.row = {
            "question": "What is the capital of France?",
            "imageLink": "https://example.com/image.jpg",
            "correctAnswer": "Paris",
            "imageFile": "image.jpg",
            "textInput": "Some text input",
        }

    @patch("benchmark.DataUtils.post_data")
    @patch("benchmark.Benchmark.process_func")
    def test_benchmark_question(self, mock_process_func, mock_post_data):
        mock_process_func.return_value = "Generated answer"

        expected_data = {
            "question": "What is the capital of France?",
            "prompt": "Create prompt from row",
            "agentType": "Agent type",
            "testType": "Test type",
            "correctAnswer": "Paris",
            "agentResponse": "Generated answer",
            "imageFile": "image.jpg",
            "imageLink": "https://example.com/image.jpg",
            "taskCategory": "",
            "multipleChoice": [],
            "metadataLink": "",
            "timestamp": "Timestamp",
            "textInput": "Some text input",
            "promptType": "Prompt type"
        }

        result = self.benchmark.benchmark_question(self.row)

        mock_process_func.assert_called_once_with("https://example.com/image.jpg", "Create prompt from row")
        mock_post_data.assert_called_once_with(expected_data)
        self.assertEqual(result, expected_data)

if __name__ == "__main__":
    unittest.main()