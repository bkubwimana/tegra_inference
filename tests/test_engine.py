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
        self.vqa_counter = 0
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
                "max_tokens": 70,
                "user_query": "Analyze the scene for object detection.",
                "priority": "quality",
                "task_complexity": "minimal",
                "response_detail": "concise",
                "time_budget": "ultra-fast"
            },
            {
                "max_tokens": 70,
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
            "max_tokens": 70,
            "user_query": query,
            "priority": random.choice(["quality", "speed"]),
            "task_complexity": random.choice(["minimal", "moderate", "complex"]),
            "response_detail": random.choice(["concise", "detailed"]),
            "time_budget": random.choice(["ultra-fast", "fast", "normal"])
        }
    def generate_all_prompt_configs(self):
        #2 x 3 x 2 x 3 factorial design
        priorities = ["quality", "speed"]
        complexities = ["minimal", "moderate", "complex"]
        details = ["concise", "detailed"]
        budgets = ["ultra-fast", "fast", "normal"]
        
        all_configs = []
        for priority in priorities:
            for complexity in complexities:
                for detail in details:
                    for budget in budgets:
                        all_configs.append({
                            "max_tokens": 70,   
                            "user_query": "",   
                            "priority": priority,
                            "task_complexity": complexity,
                            "response_detail": detail,
                            "time_budget": budget
                        })
        return all_configs

    def test_vqa_dataset_hybrid(self):
        """
        Hybrid between and within subjects:
        - For a small subset of questions, use all prompt configs (within-subject).
        - For the rest, use one random prompt config (between-subject).
        """
        if not self.vqa_data:
            self.skipTest("No VQA data available")
        
        # 1. Collect all (image, question) pairs.
        all_pairs = []
        current_image = None
        queries = []
        
        for item in self.vqa_data:
            if item == "reset":
                # End of block: store the (image, question) pairs
                if current_image and queries:
                    for q in queries:
                        all_pairs.append((current_image, q))
                # Reset for next block
                current_image = None
                queries = []
            else:
                # Check if item is path-like => treat as an image
                if item.startswith("/") or "\\" in item or (":" in item):
                    if os.path.exists(item):
                        current_image = item
                else:
                    # Otherwise it's a question
                    queries.append(item)
        
        # Edge case: if the last block never hit "reset"
        if current_image and queries:
            for q in queries:
                all_pairs.append((current_image, q))
        
        logging.info(f"Total (image, question) pairs collected: {len(all_pairs)}")
        
        # 2. Split into within-subject vs. between-subject sets.
        NUM_WITHIN_SUBJECT = 50
        random.shuffle(all_pairs)
        
        within_subject_pairs = all_pairs[:NUM_WITHIN_SUBJECT]
        between_subject_pairs = all_pairs[NUM_WITHIN_SUBJECT:201]
        
        logging.info(f"Within-subject subset size: {len(within_subject_pairs)}")
        logging.info(f"Between-subject subset size: {len(between_subject_pairs)}")
        
        # 3. prompt configurations for the within-subject test.
        all_prompt_configs = self.generate_all_prompt_configs()
        logging.info("\033[93mTotal prompt configurations: %d\033[0m", len(all_prompt_configs))
        
        output_file = os.path.join(os.path.dirname(__file__), "..", "prompt_results_hybrid.json")
        
        # Open file for immediate streaming writes
        with open(output_file, 'w') as f_out:
            # 3A. Within-Subject: stream results immediately
            logging.info("\033[93mStarting within-subject test...\033[0m")
            for (image_path, question) in within_subject_pairs:
                for config in all_prompt_configs:
                    self.prompt_engine.set_prompt_params(config)
                    with self.subTest(image=image_path, query=question, config=config):
                        self.vqa_counter += 1
                        logging.info(f"vqa #: {self.vqa_counter}")
                        response_tuple = self.prompt_engine.get_response(question, image_path=image_path)
                        formatted_output = format_response(response_tuple)
                        # Only write if we got a valid response.
                        if formatted_output:
                            f_out.write(formatted_output + "\n")
                            f_out.flush()
                        else:
                            logging.warning("\033[93mSkipping invalid response (within-subject).\033[0m")
            
            # 3B. Between-Subject: stream each result immediately
            logging.info("\033[93mStarting between-subject test...\033[0m")
            for (image_path, question) in between_subject_pairs:
                params = self.random_params(question)
                self.prompt_engine.set_prompt_params(params)
                with self.subTest(image=image_path, query=question, config=params):
                    response_tuple = self.prompt_engine.get_response(question, image_path=image_path)
                    formatted_output = format_response(response_tuple)
                    if formatted_output:
                        f_out.write(formatted_output + "\n")
                        f_out.flush()
                    else:
                        logging.warning("\033[93mSkipping invalid response (between-subject).\033[0m")
        logging.info(f"Hybrid test outputs saved to {output_file}")


if __name__ == '__main__':
    unittest.main()