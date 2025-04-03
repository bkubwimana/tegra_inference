#!/usr/bin/env python3
"""
VQA Evaluation Script for Janus Prompt Engineering Experiments

This script provides utilities to:
1. Process experiment results from prompt engineering tests
2. Convert results to standard VQA evaluation format
3. Calculate accuracy metrics by prompt parameter combinations
4. Generate comparison reports between different prompt strategies

Usage:
    python vqa_eval.py --within_results path/to/prompt_within.jsonl --between_results path/to/prompt_between.jsonl
"""

import os
import sys
import json
import argparse
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from src.output_processor import (
    extract_clean_answer, 
    convert_to_vqa_eval_format, 
    merge_with_ground_truth,
    calculate_accuracy
)

def load_jsonl_to_list(file_path):
    """
    Custom loader to parse a file containing pretty-printed JSON objects.
    Assumes that JSON objects are separated by a linebreak between a '}' and a '{'.
    """
    with open(file_path, 'r') as f:
        content = f.read().strip()
    if not content:
        return []
    # Split on the pattern where one JSON ends and another begins.
    # We remove the delimiter and then add missing braces to each block.
    blocks = content.split("}\n{")
    json_list = []
    for i, block in enumerate(blocks):
        if i == 0:
            json_str = block + "}"
        elif i == len(blocks)-1:
            json_str = "{" + block
        else:
            json_str = "{" + block + "}"
        try:
            json_list.append(json.loads(json_str))
        except json.JSONDecodeError:
            print(f"Warning: Skipping invalid JSON block: {json_str[:50]}...")
    return json_list

def compute_accuracy_metrics(model_answers, ground_truth):
    """
    Compute accuracy metrics by comparing model answers to ground truth.
    
    Args:
        model_answers: List of model answer strings
        ground_truth: List of ground truth answer strings
        
    Returns:
        Dictionary with accuracy metrics
    """
    total = len(model_answers)
    if total != len(ground_truth):
        raise ValueError(f"Number of model answers ({total}) doesn't match ground truth ({len(ground_truth)})")
    
    exact_match = 0
    fuzzy_match = 0
    
    for model_ans, gt_ans in zip(model_answers, ground_truth):
        model_ans = model_ans.lower().strip() 
        gt_ans = gt_ans.lower().strip()
        
        # Check for exact match
        if model_ans == gt_ans:
            exact_match += 1
            fuzzy_match += 1
        # Check for fuzzy match (model answer contains ground truth or vice versa)
        elif model_ans in gt_ans or gt_ans in model_ans:
            fuzzy_match += 1
    
    return {
        "total_questions": total,
        "exact_match": exact_match,
        "exact_match_accuracy": exact_match / total if total > 0 else 0,
        "fuzzy_match": fuzzy_match,
        "fuzzy_match_accuracy": fuzzy_match / total if total > 0 else 0
    }

def analyze_within_subject_results(results_data, output_file=None):
    """
    Analyze within-subject design results to compare accuracy across prompt parameters.
    
    Args:
        results_data: List of dictionaries containing results
        output_file: Optional path to save the analysis
        
    Returns:
        DataFrame with accuracy metrics by prompt parameter combinations
    """
    # Group by question to analyze performance across prompt configurations
    questions = defaultdict(list)
    for item in results_data:
        user_query = item['input_prompt'].get('user_query', '')
        clean_answer = extract_clean_answer(item['answer'])
        prompt_config = {
            'priority': item['input_prompt'].get('priority', ''),
            'task_complexity': item['input_prompt'].get('task_complexity', ''),
            'response_detail': item['input_prompt'].get('response_detail', ''),
            'time_budget': item['input_prompt'].get('time_budget', '')
        }
        questions[user_query].append({
            'answer': clean_answer,
            'config': prompt_config,
            'token_count': item.get('token_count', 0)
        })
    
    # Prepare data for analysis
    analysis_data = []
    
    # For now we don't have ground truth so we'll just analyze answer consistency
    for question, answers in questions.items():
        # Group by config parameters
        config_groups = defaultdict(list)
        
        for ans_item in answers:
            config_key = (
                ans_item['config']['priority'],
                ans_item['config']['task_complexity'],
                ans_item['config']['response_detail'],
                ans_item['config']['time_budget']
            )
            config_groups[config_key].append(ans_item)
        
        # Analyze each config group
        for config_key, items in config_groups.items():
            priority, task_complexity, response_detail, time_budget = config_key
            
            # Calculate metrics
            token_counts = [item['token_count'] for item in items]
            avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0
            
            analysis_data.append({
                'question': question,
                'priority': priority,
                'task_complexity': task_complexity,
                'response_detail': response_detail,
                'time_budget': time_budget,
                'answers': [item['answer'] for item in items],
                'avg_token_count': avg_tokens,
                'sample_count': len(items)
            })
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(analysis_data)
    
    # Analyze by parameter combinations
    param_analysis = []
    
    # All parameter combinations
    param_combinations = [
        ('priority',),
        ('task_complexity',),
        ('response_detail',),
        ('time_budget',),
        ('priority', 'task_complexity'),
        ('priority', 'response_detail'),
        ('priority', 'time_budget'),
        ('task_complexity', 'response_detail'),
        ('task_complexity', 'time_budget'),
        ('response_detail', 'time_budget'),
    ]
    
    for params in param_combinations:
        grouped = df.groupby(list(params))
        
        for group_key, group_df in grouped:
            if not isinstance(group_key, tuple):
                group_key = (group_key,)
            
            param_dict = {param: value for param, value in zip(params, group_key)}
            avg_tokens = group_df['avg_token_count'].mean()
            
            # Since we don't have ground truth yet, we'll just report token stats
            param_analysis.append({
                **param_dict,
                'parameter_combo': '+'.join(params),
                'avg_token_count': avg_tokens,
                'sample_count': len(group_df)
            })
    
    # Convert to DataFrame
    analysis_df = pd.DataFrame(param_analysis)
    
    # Save if output file specified
    if output_file:
        analysis_df.to_csv(output_file, index=False)
    
    return analysis_df

def analyze_between_subject_results(results_data, output_file=None):
    """
    Analyze between-subject design results to compare accuracy across prompt parameters.
    
    Args:
        results_data: List of dictionaries containing results
        output_file: Optional path to save the analysis
        
    Returns:
        DataFrame with accuracy metrics by prompt parameter combinations
    """
    # Extract relevant data
    analysis_data = []
    
    for item in results_data:
        clean_answer = extract_clean_answer(item['answer'])
        analysis_data.append({
            'question': item['input_prompt'].get('user_query', ''),
            'priority': item['input_prompt'].get('priority', ''),
            'task_complexity': item['input_prompt'].get('task_complexity', ''),
            'response_detail': item['input_prompt'].get('response_detail', ''),
            'time_budget': item['input_prompt'].get('time_budget', ''),
            'answer': clean_answer,
            'token_count': item.get('token_count', 0)
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(analysis_data)
    
    # Analyze by parameter combinations (similar to within-subject)
    param_analysis = []
    
    # All parameter combinations
    param_combinations = [
        ('priority',),
        ('task_complexity',),
        ('response_detail',),
        ('time_budget',),
        ('priority', 'task_complexity'),
        ('priority', 'response_detail'),
        ('priority', 'time_budget'),
        ('task_complexity', 'response_detail'),
        ('task_complexity', 'time_budget'),
        ('response_detail', 'time_budget'),
    ]
    
    for params in param_combinations:
        grouped = df.groupby(list(params))
        
        for group_key, group_df in grouped:
            if not isinstance(group_key, tuple):
                group_key = (group_key,)
            
            param_dict = {param: value for param, value in zip(params, group_key)}
            avg_tokens = group_df['token_count'].mean()
            
            param_analysis.append({
                **param_dict,
                'parameter_combo': '+'.join(params),
                'avg_token_count': avg_tokens,
                'sample_count': len(group_df)
            })
    
    # Convert to DataFrame
    analysis_df = pd.DataFrame(param_analysis)
    
    # Save if output file specified
    if output_file:
        analysis_df.to_csv(output_file, index=False)
    
    return analysis_df

def visualize_parameter_impact(within_df, between_df, output_dir):
    """
    Create visualizations showing the impact of different prompt parameters on performance.
    
    Args:
        within_df: DataFrame with within-subject analysis
        between_df: DataFrame with between-subject analysis
        output_dir: Directory to save the visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create combined chart for token counts
    single_params = [
        'priority', 'task_complexity', 'response_detail', 'time_budget'
    ]
    
    for param in single_params:
        plt.figure(figsize=(12, 6))
        
        # Setup for combined plot
        within_data = within_df[within_df['parameter_combo'] == param]
        between_data = between_df[between_df['parameter_combo'] == param]
        
        # Create bar chart
        x = range(len(within_data))
        width = 0.35
        
        # Plot within data
        plt.bar(
            [i - width/2 for i in x], 
            within_data['avg_token_count'], 
            width=width, 
            label='Within-Subject'
        )
        
        # Plot between data
        plt.bar(
            [i + width/2 for i in x], 
            between_data['avg_token_count'], 
            width=width, 
            label='Between-Subject'
        )
        
        # Configure chart
        plt.xlabel(param)
        plt.ylabel('Average Token Count')
        plt.title(f'Impact of {param} on Response Length')
        plt.xticks(x, within_data[param])
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Save
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{param}_token_comparison.png'))
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Analyze VQA experiment results')
    parser.add_argument('--within_results', help='Path to within-subject results file')
    parser.add_argument('--between_results', help='Path to between-subject results file')
    parser.add_argument('--output_dir', default='../analysis_results', help='Output directory for analysis')
    parser.add_argument('--ground_truth', help='Optional path to ground truth annotations')
    args = parser.parse_args()
    
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Load results
    if args.within_results:
        within_data = load_jsonl_to_list(args.within_results)
        print(f"Loaded {len(within_data)} results from within-subject experiment")
        
        within_analysis = analyze_within_subject_results(
            within_data, 
            os.path.join(output_dir, 'within_analysis.csv')
        )
        print(f"Completed within-subject analysis")
    else:
        within_data = []
        within_analysis = pd.DataFrame()
    
    if args.between_results:
        between_data = load_jsonl_to_list(args.between_results)
        print(f"Loaded {len(between_data)} results from between-subject experiment")
        
        between_analysis = analyze_between_subject_results(
            between_data, 
            os.path.join(output_dir, 'between_analysis.csv')
        )
        print(f"Completed between-subject analysis")
    else:
        between_data = []
        between_analysis = pd.DataFrame()
    
    # Create visualizations if both datasets are available
    if not within_analysis.empty and not between_analysis.empty:
        print("Creating comparative visualizations...")
        visualize_parameter_impact(
            within_analysis, 
            between_analysis, 
            os.path.join(output_dir, 'charts')
        )
    
    # Process for VQA evaluation format if ground truth is available
    if args.ground_truth and (args.within_results or args.between_results):
        print("Converting results to VQA evaluation format...")
        
        if args.within_results:
            within_vqa_format = convert_to_vqa_eval_format(
                args.within_results,
                os.path.join(output_dir, 'within_vqa_format.json')
            )
        
        if args.between_results:
            between_vqa_format = convert_to_vqa_eval_format(
                args.between_results,
                os.path.join(output_dir, 'between_vqa_format.json')
            )
        
        print("Results converted to VQA format for use with evaluation APIs")
    
    print(f"Analysis complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main()