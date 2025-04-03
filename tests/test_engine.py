import os
import json
import unittest
import random
import sys
import logging
from pathlib import Path

# Configure logging to display time and log level
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

src_path = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.append(src_path)
from prompt_engine import PromptEngine
from output_processor import format_response, save_output_to_file

# Import loadset functions
from loadset import download_and_extract, load_vqa_subset

class TestPromptEngine(unittest.TestCase):

    def setUp(self):
        logging.info("Initializing PromptEngine for tests.")
        self.prompt_engine = PromptEngine()
        self.vqa_counter = 0
        
        # Load VQA dataset using loadset.py functions
        try:
            logging.info("Loading VQA dataset from loadset module...")
            download_and_extract()  # Ensure data is downloaded
            self.vqa_examples = load_vqa_subset()
            logging.info(f"Loaded {len(self.vqa_examples)} VQA examples")
        except Exception as e:
            logging.error(f"Error loading VQA dataset: {e}")
            self.vqa_examples = []

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
        Hybrid between and within subjects using the VQA dataset loaded through loadset.py.
        - For a small subset of questions, use all prompt configs (within-subject).
        - For the rest, use one random prompt config (between-subject).
        """
        if not self.vqa_examples:
            self.skipTest("No VQA data available")
        
        # 1. Collect all (image, question, question_id) tuples
        all_examples = []
        for example in self.vqa_examples:
            image_path = example['image_file_path']
            question = example['question']
            question_id = example['question_id']
            all_examples.append((image_path, question, question_id))
        
        logging.info(f"Total examples collected: {len(all_examples)}")
        
        # 2. Split into within-subject and between-subject sets
        # Using a smaller set for within-subject since it multiplies with all prompt configurations
        within_subject_examples = all_examples[:20]  # Reduced from 100 to 20 for manageable testing
        between_subject_examples = all_examples[20:100]
        
        logging.info(f"Within-subject subset size: {len(within_subject_examples)}")
        logging.info(f"Between-subject subset size: {len(between_subject_examples)}")
        
        # 3. Generate prompt configurations
        all_prompt_configs = self.generate_all_prompt_configs()
        logging.info("\033[93mTotal prompt configurations: %d\033[0m", len(all_prompt_configs))
        
        within_file = os.path.join(os.path.dirname(__file__), "..", "prompt_within.jsonl")
        between_file = os.path.join(os.path.dirname(__file__), "..", "prompt_between.jsonl")
        
        skipped_count = 0
        
        # Stream writing within-subject results
        logging.info("\033[93mStarting within-subject test...\033[0m")
        with open(within_file, 'w') as f_within:
            for (image_path, question, question_id) in within_subject_examples:
                for config in all_prompt_configs:
                    config["user_query"] = question  # Set the query in the config
                    self.prompt_engine.set_prompt_params(config)
                    with self.subTest(image=image_path, query=question, config=config):
                        self.vqa_counter += 1
                        # Remove metadata parameter and just pass image_path
                        response_tuple = self.prompt_engine.get_response(
                            question, 
                            image_path=image_path
                        )
                        
                        # Add question_id to the response tuple after the call returns
                        response_with_id = response_tuple + (question_id,)
                        formatted_output = format_response(response_with_id)
                        
                        if formatted_output:
                            output_json = json.loads(formatted_output)
                            if output_json.get("token_count", 0) == 512:
                                logging.warning("\033[91mSkipping response with token_count 512 (within-subject).\033[0m")
                                continue
                            f_within.write(formatted_output + "\n")
                        else:
                            logging.warning("Skipping invalid response (within-subject).")
        
        # Stream writing between-subject results
        logging.info("\033[93mStarting between-subject test...\033[0m")
        with open(between_file, 'w') as f_between:
            for (image_path, question, question_id) in between_subject_examples:
                params = self.random_params(question)
                self.prompt_engine.set_prompt_params(params)
                with self.subTest(image=image_path, query=question, config=params):
                    # Remove metadata parameter and just pass image_path
                    response_tuple = self.prompt_engine.get_response(
                        question, 
                        image_path=image_path
                    )
                    
                    # Add question_id to the response tuple after the call returns
                    response_with_id = response_tuple + (question_id,)
                    formatted_output = format_response(response_with_id)
                    
                    if formatted_output:
                        output_json = json.loads(formatted_output)
                        if output_json.get("token_count", 0) == 512:
                            logging.warning("\033[91mSkipping response with token_count 512 (between-subject).\033[0m")
                            continue
                        f_between.write(formatted_output + "\n")
                    else:
                        logging.warning("\033[93mSkipping invalid response (between-subject).\033[0m")
                        skipped_count += 1
                        
        logging.info(f"Skipped {skipped_count} invalid responses during between-subject test.")
        logging.info(f"Streamed within-subject results to: {within_file}")
        logging.info(f"Streamed between-subject results to: {between_file}")


if __name__ == '__main__':
    unittest.main()