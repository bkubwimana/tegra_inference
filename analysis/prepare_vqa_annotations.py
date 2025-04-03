#!/usr/bin/env python3
"""
Prepare VQA Annotations for Evaluation

This script processes the VQA dataset to extract question-answer pairs and prepares
them as ground truth annotations for evaluating model predictions.

The script performs the following steps:
1. Loads VQA dataset annotations (from the VQA val subset)
2. Extracts questions and ground truth answers
3. Creates mappings between questions/images and their annotations
4. Saves the annotations in a format that can be used with the VQA evaluation script

Usage:
    python prepare_vqa_annotations.py --vqa_dataset_path /path/to/vqa/dataset --output_file annotations.json
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

def extract_vqa_annotations(dataset_path, output_file=None):
    """
    Extract VQA annotations from the dataset.
    
    Args:
        dataset_path: Path to the VQA dataset
        output_file: Path to save the extracted annotations
        
    Returns:
        Dictionary with annotations if output_file is None,
        otherwise saves to the specified file
    """
    annotations_path = os.path.join(dataset_path, "v2_mscoco_val2014_annotations.json")
    questions_path = os.path.join(dataset_path, "v2_OpenEnded_mscoco_val2014_questions.json")
    
    if not os.path.exists(annotations_path) or not os.path.exists(questions_path):
        raise FileNotFoundError(f"VQA dataset files not found at {dataset_path}")
    
    # Load annotations and questions
    with open(annotations_path, 'r') as f:
        annotation_data = json.load(f)
    
    with open(questions_path, 'r') as f:
        question_data = json.load(f)
    
    # Extract questions
    questions = {q['question_id']: q for q in question_data['questions']}
    
    # Process annotations
    processed_annotations = []
    
    for ann in annotation_data['annotations']:
        question_id = ann['question_id']
        if question_id in questions:
            # Combine question and annotation information
            processed_ann = {
                'question_id': question_id,
                'image_id': ann['image_id'],
                'question': questions[question_id]['question'],
                'multiple_choice_answer': ann['multiple_choice_answer'],
                'answers': ann['answers']
            }
            processed_annotations.append(processed_ann)
    
    # Create mappings for easy lookup
    question_to_annotation = {}
    image_question_to_annotation = {}
    
    for ann in processed_annotations:
        question_text = ann['question'].strip().lower()
        question_to_annotation[question_text] = ann
        
        # Create a compound key of image_id+question for cases where the same question
        # appears with different images
        image_question_key = f"{ann['image_id']}_{question_text}"
        image_question_to_annotation[image_question_key] = ann
    
    result = {
        'annotations': processed_annotations,
        'question_to_annotation': question_to_annotation,
        'image_question_to_annotation': image_question_to_annotation
    }
    
    # Save to file if requested
    if output_file:
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Annotations saved to {output_file}")
    
    return result

def create_question_id_mapping(annotations):
    """
    Create a mapping from question text to question ID.
    
    Args:
        annotations: Dictionary with annotations
        
    Returns:
        Dictionary mapping question text to question ID
    """
    mapping = {}
    
    for ann in annotations['annotations']:
        question_text = ann['question'].strip().lower()
        mapping[question_text] = ann['question_id']
    
    return mapping

def main():
    parser = argparse.ArgumentParser(description='Prepare VQA annotations for evaluation')
    parser.add_argument('--vqa_dataset_path', required=True, help='Path to the VQA dataset')
    parser.add_argument('--output_file', default='../data/vqa_annotations.json', help='Output file for annotations')
    parser.add_argument('--mapping_file', default='../data/question_id_mapping.json', help='Output file for question ID mapping')
    args = parser.parse_args()
    
    try:
        # Extract annotations
        annotations = extract_vqa_annotations(args.vqa_dataset_path, args.output_file)
        print(f"Extracted {len(annotations['annotations'])} annotations")
        
        # Create and save question ID mapping
        mapping = create_question_id_mapping(annotations)
        
        os.makedirs(os.path.dirname(os.path.abspath(args.mapping_file)), exist_ok=True)
        with open(args.mapping_file, 'w') as f:
            json.dump(mapping, f, indent=2)
        
        print(f"Question ID mapping saved to {args.mapping_file}")
        print(f"Created mappings for {len(mapping)} questions")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()