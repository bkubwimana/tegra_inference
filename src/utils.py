import os
import json
import time

def load_vqa_data(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def measure_latency(start_time):
    return time.time() - start_time

def preprocess_image(image_path):
    # Placeholder for image preprocessing logic
    pass

def format_output(response):
    # Placeholder for output formatting logic
    return response.strip()