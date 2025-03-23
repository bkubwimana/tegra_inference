import json
import re

REPEATED_FIELDS = [
    "User Query", "Priority", "Output Range", "Task Complexity", "Response Detail", "Time Budget"
]

def format_response(response_tuple: tuple) -> str:
    """
    Receives a tuple (formatted_prompt, answer, total_latency, generation_latency, decode_latency, token_count)
    and returns a JSON string with the structured data.
    If the answer is determined to be 'messed up' by repeating prompt elements, returns None.
    """
    formatted_prompt, answer, total_latency, generation_latency, decode_latency, token_count = response_tuple

    # Ensure answer has BEGIN/END delimiters
    if "BEGIN_RESPONSE" not in answer or "END_RESPONSE" not in answer:
        answer = f"BEGIN_RESPONSE\n{answer.strip()}\nEND_RESPONSE"
    
    # Check if answer is messed up by having repeated prompt markers.
    # For each field, if its pattern appears more than once, we consider it messed up.
    for field in REPEATED_FIELDS:
        if answer.count(f"[{field}:") > 1:
            # Skip this response entirely by returning None.
            return None
    
    # Use regular expressions to extract dynamic elements from the formatted prompt
    user_query = re.search(r"USER_QUERY: (.*)", formatted_prompt).group(1)
    priority = re.search(r"PRIORITY: (.*)", formatted_prompt).group(1)
    min_tokens = re.search(r"OUTPUT_RANGE: between (\d+) and \d+ tokens", formatted_prompt).group(1)
    max_tokens = re.search(r"OUTPUT_RANGE: between \d+ and (\d+) tokens", formatted_prompt).group(1)
    task_complexity = re.search(r"TASK_COMPLEXITY: (.*)", formatted_prompt).group(1)
    response_detail = re.search(r"RESPONSE_DETAIL: (.*)", formatted_prompt).group(1)
    time_budget = re.search(r"TIME_BUDGET: (.*)", formatted_prompt).group(1)

    output = {
        "input_prompt": {
            "user_query": user_query,
            "priority": priority,
            "min_tokens": min_tokens,
            "max_tokens": max_tokens,
            "task_complexity": task_complexity,
            "response_detail": response_detail,
            "time_budget": time_budget
        },
        "answer": answer,
        "latency": {
            "total": total_latency,
            "generation": generation_latency,
            "decode": decode_latency
        },
        "token_count": token_count
    }
    return json.dumps(output, indent=2)

def free_format_response(response_tuple: tuple) -> str:
    """
    Receives a tuple (raw_query, answer, total_latency, generation_latency, decode_latency, token_count)
    and returns a JSON string with the structured data.
    This uses a simple format without extracting dynamic prompt elements.
    """
    raw_query, answer, total_latency, generation_latency, decode_latency, token_count = response_tuple

    # Ensure answer includes BEGIN/END delimiters
    if "BEGIN_RESPONSE" not in answer or "END_RESPONSE" not in answer:
        answer = f"BEGIN_RESPONSE\n{answer.strip()}\nEND_RESPONSE"
    
    output = {
        "input_prompt": {
            "user_query": raw_query,
        },
        "answer": answer,
        "latency": {
            "total": total_latency,
            "generation": generation_latency,
            "decode": decode_latency
        },
        "token_count": token_count
    }
    return json.dumps(output, indent=2)


def process_vqa_output_json(model_output):
    """
    Processes a list of output tuples and returns a list of JSON formatted strings,
    skipping those responses identified as messed up.
    """
    processed_output = []
    for item in model_output:
        formatted = format_response(item)
        if formatted is not None:
            processed_output.append(formatted)
    return processed_output

def save_output_to_file(output, file_path: str):
    with open(file_path, 'w') as f:
        f.write('\n'.join(output))