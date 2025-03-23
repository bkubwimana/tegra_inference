import os
import json
import sys
import logging
import statistics

# Ensure our src directory is in the path.
import sys
src_path = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.append(src_path)

from free_engine import FreePromptEngine
from output_processor import free_format_response

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

vqa_data_path = os.path.join(os.path.dirname(__file__), "..", "10_percent_vqa.json")

def run_free_test(output_file):
    engine = FreePromptEngine()
    
    if not os.path.exists(vqa_data_path):
        logging.error(f"VQA data file not found: {vqa_data_path}")
        sys.exit(1)
        
    with open(vqa_data_path, 'r') as f:
        vqa_data = json.load(f)
    
    results = []
    latencies = []
    token_counts = []
    
    current_image = None  # Holds the current image path, if any.
    
    for item in vqa_data:
        # Clean up the item
        item = item.strip()
        
        # If a "reset" is encountered, clear the current image.
        if item.lower() == "reset":
            current_image = None
            continue
        
        # If the item ends with typical image extensions, set it as the image.
        elif item.lower().endswith((".jpg", ".png", ".jpeg")):
            current_image = item
            continue
        
        # Otherwise, assume it's a question.
        else:
            question = item
            # logging.info(f"Processing question: {question}")
            # Pass the current image (could be None) along with the question.
            response_tuple = engine.get_response(question, image_path=current_image)
            formatted = free_format_response(response_tuple)
            if formatted is None:
                logging.info("Skipping messed up response.")
                continue
            result_json = json.loads(formatted)
            results.append(result_json)
            
            # Collect latency and token count data
            if "latency" in result_json and "total" in result_json["latency"]:
                latencies.append(result_json["latency"]["total"])
            if "token_count" in result_json:
                token_counts.append(result_json["token_count"])
    
    # Compute summary statistics if we got any valid responses
    summary = {}
    if latencies:
        summary["avg_latency"] = statistics.mean(latencies)
        summary["min_latency"] = min(latencies)
        summary["max_latency"] = max(latencies)
    if token_counts:
        summary["avg_token_count"] = statistics.mean(token_counts)
        summary["min_token_count"] = min(token_counts)
        summary["max_token_count"] = max(token_counts)
    summary["total_tests"] = len(results)
    
    output_data = {
        "summary": summary,
        "results": results
    }
    
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
        
    logging.info(f"Free test completed. Results saved to {output_file}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python free_test.py<output_results.json>")
        sys.exit(1)
    output_file = sys.argv[1]
    
    run_free_test(output_file)

if __name__ == "__main__":
    main()