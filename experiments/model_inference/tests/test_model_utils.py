import unittest
from unittest.mock import MagicMock
from model_utils import ModelUtils

class TestModelUtils(unittest.TestCase):
    def setUp(self):
        self.model_utils = ModelUtils("Salesforce/blip2-opt-2.7b")

    def test_process_image_with_blip2_model(self):
        # Mock the necessary dependencies
        self.model_utils.load_blip2 = MagicMock(return_value=(MagicMock(), MagicMock()))
        self.model_utils.processor = MagicMock()
        self.model_utils.processor.batch_decode = MagicMock(return_value=["Generated Text"])

        # Call the method under test
        result = self.model_utils.process_image("image_url", "prompt")

        # Assert the result
        self.assertEqual(result, "Generated Text")

    def test_process_image_with_gpt_4v_model(self):
        # Mock the necessary dependencies
        self.model_utils.load_blip2 = MagicMock()
        self.model_utils.processor = MagicMock()
        self.model_utils.processor.batch_decode = MagicMock()
        self.model_utils.model = MagicMock()
        self.model_utils.model.generate = MagicMock(return_value=["Generated Text"])

        # Call the method under test
        result = self.model_utils.process_image("image_url", "prompt", max_length=270)

        # Assert the result
        self.assertEqual(result, "Generated Text")

    def test_process_image_with_default_model(self):
        # Mock the necessary dependencies
        self.model_utils.load_blip2 = MagicMock()
        self.model_utils.load_llava = MagicMock(return_value=(MagicMock(), MagicMock()))
        self.model_utils.processor = MagicMock()
        self.model_utils.processor.batch_decode = MagicMock(return_value=["Generated Text"])

        # Call the method under test
        result = self.model_utils.process_image("image_url", "prompt")

        # Assert the result
        self.assertEqual(result, "Generated Text")

    def test_load_blip2(self):
        # Call the method under test
        model, processor = self.model_utils.load_blip2()

        # Assert the result
        self.assertIsNotNone(model)
        self.assertIsNotNone(processor)

    def test_load_llava(self):
        # Call the method under test
        model, processor = self.model_utils.load_llava()

        # Assert the result
        self.assertIsNotNone(model)
        self.assertIsNotNone(processor)

if __name__ == "__main__":
    unittest.main()