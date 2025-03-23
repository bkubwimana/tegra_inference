#### language: python
# filepath: /mnt/packages/models/deepseek/janus/tests/test_engine.py
import os
import json
import unittest
import random
import sys
import logging

# Configure logging to display time and log level
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

src_path = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.append(src_path)
from prompt_engine import PromptEngine
from output_processor import format_response, save_output_to_file

class TestPromptEngine(unittest.TestCase):

    def setUp(self):
        logging.info("Initializing PromptEngine for tests.")
        self.prompt_engine = PromptEngine()
        # Load VQA dataset (adjust path as needed)
        vqa_data_path = os.path.join(os.path.dirname(__file__), "..", "10_percent_vqa.json")
        if os.path.exists(vqa_data_path):
            logging.info(f"Loading VQA data from {vqa_data_path}")
            with open(vqa_data_path, 'r') as f:
                self.vqa_data = json.load(f)
        else:
            logging.info("No VQA data found.")
            self.vqa_data = []

    def test_dynamic_variations(self):
        variations = [
            {
                "min_tokens": 20,
                "max_tokens": 50,
                "user_query": "Analyze the scene for object detection.",
                "priority": "quality",
                "task_complexity": "minimal",
                "response_detail": "concise",
                "time_budget": "ultra-fast"
            },
            {
                "min_tokens": 30,
                "max_tokens": 60,
                "user_query": "What is in the background?",
                "priority": "speed",
                "task_complexity": "moderate",
                "response_detail": "detailed",
                "time_budget": "fast"
            }
        ]
        for params in variations:
            with self.subTest(params=params):
                logging.info(f"Testing dynamic variation with params: {params}")
                self.prompt_engine.set_prompt_params(params)
                response_tuple = self.prompt_engine.get_response(params["user_query"])
                formatted_output = format_response(response_tuple)
                logging.info(f"Response: {formatted_output}")

                # Parse the JSON string to a dict
                output_json = json.loads(formatted_output)

                # Assert presence of keys
                self.assertIn("input_prompt", output_json)
                self.assertIn("answer", output_json)
                self.assertIn("latency", output_json)
                self.assertIn("token_count", output_json)

                answer = output_json["answer"]
                # Verify that the answer contains delimiters
                self.assertIn("BEGIN_RESPONSE", answer)
                self.assertIn("END_RESPONSE", answer)

                # Extract only the answer text between the delimiters
                begin_index = answer.find("BEGIN_RESPONSE")
                end_index = answer.find("END_RESPONSE", begin_index)
                # Ensure valid indices
                self.assertGreater(begin_index, -1)
                self.assertGreater(end_index, -1)
                answer_text = answer[begin_index+len("BEGIN_RESPONSE"):end_index].strip()

                token_count = output_json["token_count"]
                # self.assertGreaterEqual(token_count, params["min_tokens"])
                # self.assertLessEqual(token_count, params["max_tokens"])
    def random_params(self, query):
        return {
            "min_tokens": random.randint(20, 30),
            "max_tokens": random.randint(40, 60),
            "user_query": query,
            "priority": random.choice(["quality", "speed"]),
            "task_complexity": random.choice(["minimal", "moderate", "complex"]),
            "response_detail": random.choice(["concise", "detailed"]),
            "time_budget": random.choice(["ultra-fast", "fast", "normal"])
        }
    def test_vqa_dataset(self):
        if not self.vqa_data:
            self.skipTest("No VQA data available")
        output_results = []
        current_image = None
        queries = []
        for item in self.vqa_data:
            if item == "reset":
                if current_image and queries:
                    for query in queries:
                        params = self.random_params(query)
                        self.prompt_engine.set_prompt_params(params)
                        with self.subTest(image=current_image, query=query):
                            # logging.info(f"Processing query '{query}' for image '{current_image}'")
                            response_tuple = self.prompt_engine.get_response(query, image_path=current_image)
                            formatted_output = format_response(response_tuple)
                            # logging.info(f"Response: {formatted_output}")
                            
                            # Parse the JSON string to a dict
                            output_json = json.loads(formatted_output)
                            
                            # Assert structured keys are present
                            self.assertIn("input_prompt", output_json)
                            self.assertIn("answer", output_json)
                            self.assertIn("latency", output_json)
                            self.assertIn("token_count", output_json)
                            
                            # Check that answer contains delimiters
                            answer = output_json["answer"]
                            self.assertIn("BEGIN_RESPONSE", answer)
                            self.assertIn("END_RESPONSE", answer)
                            
                            output_results.append(formatted_output)
                current_image = None
                queries = []
            else:
                # Heuristic: determine whether the item is an image path
                if item.startswith("/") or item.startswith("\\") or (":" in item):
                    if os.path.exists(item):
                        current_image = item
                else:
                    queries.append(item)
        output_file = os.path.join(os.path.dirname(__file__), "..", "prompt_results.txt")
        save_output_to_file(output_results, output_file)
        logging.info(f"VQA test outputs saved to {output_file}")

if __name__ == '__main__':
    unittest.main()
