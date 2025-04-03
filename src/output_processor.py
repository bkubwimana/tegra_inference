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
    Optional question_id can be passed as part of the tuple for VQA evaluation.
    """
    # Check if the tuple has a question_id (7 elements instead of 6)
    if len(response_tuple) == 7:
        formatted_prompt, answer, total_latency, generation_latency, decode_latency, token_count, question_id = response_tuple
    else:
        formatted_prompt, answer, total_latency, generation_latency, decode_latency, token_count = response_tuple
        question_id = None

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
    max_tokens = re.search(r"OUTPUT_RANGE: up to (\d+) tokens", formatted_prompt).group(1)
    task_complexity = re.search(r"TASK_COMPLEXITY: (.*)", formatted_prompt).group(1)
    response_detail = re.search(r"RESPONSE_DETAIL: (.*)", formatted_prompt).group(1)
    time_budget = re.search(r"TIME_BUDGET: (.*)", formatted_prompt).group(1)

    output = {
        "input_prompt": {
            "user_query": user_query,
            "priority": priority,
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
    
    # Add question_id if available
    if question_id is not None:
        output["question_id"] = question_id
    
    return json.dumps(output, indent=2)

def free_format_response(response_tuple: tuple) -> str:
    """
    Receives a tuple (raw_query, answer, total_latency, generation_latency, decode_latency, token_count)
    and returns a JSON string with the structured data.
    This uses a simple format without extracting dynamic prompt elements.
    Optional question_id can be passed as part of the tuple for VQA evaluation.
    """
    # Check if the tuple has a question_id (7 elements instead of 6)
    if len(response_tuple) == 7:
        raw_query, answer, total_latency, generation_latency, decode_latency, token_count, question_id = response_tuple
    else:
        raw_query, answer, total_latency, generation_latency, decode_latency, token_count = response_tuple
        question_id = None

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
    
    # Add question_id if available
    if question_id is not None:
        output["question_id"] = question_id
        
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

# New functions for VQA evaluation compatibility

def extract_clean_answer(answer_text):
    """
    Extracts the clean answer text from the BEGIN_RESPONSE/END_RESPONSE format,
    removing any square brackets or other formatting.
    """
    # First extract the text between BEGIN_RESPONSE and END_RESPONSE
    if "BEGIN_RESPONSE" in answer_text and "END_RESPONSE" in answer_text:
        begin_idx = answer_text.find("BEGIN_RESPONSE") + len("BEGIN_RESPONSE")
        end_idx = answer_text.find("END_RESPONSE")
        answer_text = answer_text[begin_idx:end_idx].strip()
    
    # Remove any [Your Answer:] or similar prefixes
    answer_text = re.sub(r'\[\s*(?:Your Answer:|Answer:|Response:)?\s*(.*?)\s*\]', r'\1', answer_text, flags=re.DOTALL)
    
    # Clean up any remaining brackets and extra whitespace
    answer_text = re.sub(r'\[\s*(.*?)\s*\]', r'\1', answer_text, flags=re.DOTALL)
    answer_text = answer_text.strip()
    
    return answer_text

def convert_to_vqa_eval_format(jsonl_file, output_file=None, question_id_mapping=None):
    """
    Converts our custom JSONL format to the standard VQA evaluation format.
    
    Args:
        jsonl_file: Path to JSONL file with our custom formatted results
        output_file: Path to save the VQA-compatible output (if None, returns the data)
        question_id_mapping: Dictionary mapping question text to question IDs (if available)
    
    Returns:
        List of dictionaries in VQA evaluation format if output_file is None,
        otherwise saves to the specified file
    """
    results = []
    
    with open(jsonl_file, 'r') as f:
        for line in f:
            if not line.strip():
                continue
                
            data = json.loads(line)
            
            # Extract the query and clean answer
            query = data.get('input_prompt', {}).get('user_query', '')
            raw_answer = data.get('answer', '')
            clean_answer = extract_clean_answer(raw_answer)
            
            # Create an entry in VQA format
            result_entry = {
                "question": query,
                "answer": clean_answer
            }
            
            # Add question_id if available in the mapping
            if question_id_mapping and query in question_id_mapping:
                result_entry["question_id"] = question_id_mapping[query]
            
            results.append(result_entry)
    
    # Save to file if requested
    if output_file:
        with open(output_file, 'w') as f:
            json.dump({"results": results}, f, indent=2)
        return None
    
    return {"results": results}

def merge_with_ground_truth(results_file, annotations_file, output_file=None):
    """
    Merges model prediction results with ground truth annotations for evaluation.
    
    Args:
        results_file: Path to results file in VQA format
        annotations_file: Path to ground truth annotations file
        output_file: Path to save the merged data (if None, returns the data)
    
    Returns:
        Dictionary with merged data if output_file is None,
        otherwise saves to the specified file
    """
    # Load results and annotations
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)
    
    # Create a lookup for annotations by question_id
    if 'annotations' in annotations:
        annotation_lookup = {ann['question_id']: ann for ann in annotations['annotations']}
    else:
        annotation_lookup = {ann['question_id']: ann for ann in annotations}
    
    # Merge results with ground truth
    merged_data = []
    for result in results['results']:
        if 'question_id' in result and result['question_id'] in annotation_lookup:
            ann = annotation_lookup[result['question_id']]
            merged_entry = {
                "question_id": result['question_id'],
                "question": result.get('question', ann.get('question', '')),
                "image_id": ann.get('image_id', ''),
                "model_answer": result['answer'],
                "ground_truth": {
                    "multiple_choice_answer": ann.get('multiple_choice_answer', ''),
                    "answers": ann.get('answers', [])
                }
            }
            merged_data.append(merged_entry)
    
    # Save to file if requested
    if output_file:
        with open(output_file, 'w') as f:
            json.dump({"results": merged_data}, f, indent=2)
        return None
    
    return {"results": merged_data}

def calculate_accuracy(merged_data_file, output_file=None):
    """
    Calculate accuracy metrics for VQA evaluation.
    
    Args:
        merged_data_file: Path to file with merged results and ground truth
        output_file: Path to save the accuracy metrics (if None, returns the metrics)
    
    Returns:
        Dictionary with accuracy metrics if output_file is None,
        otherwise saves to the specified file
    """
    # Load merged data
    with open(merged_data_file, 'r') as f:
        data = json.load(f)
    
    results = data['results']
    total = len(results)
    exact_match = 0
    fuzzy_match = 0
    
    for item in results:
        model_answer = item['model_answer'].lower().strip()
        gt_answer = item['ground_truth']['multiple_choice_answer'].lower().strip()
        
        # Check for exact match
        if model_answer == gt_answer:
            exact_match += 1
            fuzzy_match += 1
        # Check for fuzzy match (model answer contains ground truth or vice versa)
        elif model_answer in gt_answer or gt_answer in model_answer:
            fuzzy_match += 1
    
    metrics = {
        "total_questions": total,
        "exact_match": exact_match,
        "exact_match_accuracy": exact_match / total if total > 0 else 0,
        "fuzzy_match": fuzzy_match,
        "fuzzy_match_accuracy": fuzzy_match / total if total > 0 else 0
    }
    
    # Save to file if requested
    if output_file:
        with open(output_file, 'w') as f:
            json.dump({"metrics": metrics}, f, indent=2)
        return None
    
    return {"metrics": metrics}