#!/usr/bin/env python3
"""
End-to-End VQA Evaluation Workflow

This script runs a complete evaluation workflow for VQA experiments:
1. Processes prompt engineering experiment results
2. Extracts clean answers from model outputs
3. Compares against ground truth VQA annotations
4. Calculates accuracy metrics by prompt parameters
5. Generates visualizations and reports

Usage:
    python run_vqa_evaluation.py \
        --within_results ../prompt_within.jsonl \
        --between_results ../prompt_between.jsonl \
        --output_dir ../evaluation_results
"""
import os
import sys
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict 

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from tests.loadset import load_vqa_subset  

from src.output_processor import (
    extract_clean_answer,
    convert_to_vqa_eval_format,
    merge_with_ground_truth,
    calculate_accuracy
)
from vqa_eval import (
    load_jsonl_to_list,
    analyze_within_subject_results,
    analyze_between_subject_results,
    visualize_parameter_impact
)
from prepare_vqa_annotations import (
    extract_vqa_annotations,
    create_question_id_mapping
)

def calculate_param_metrics(results, annotations, param_field):
    """
    Calculate accuracy metrics grouped by a specific prompt parameter.
    
    Args:
        results: List of result dictionaries
        annotations: Dictionary with VQA annotations
        param_field: Parameter field to group by (e.g., 'time_budget')
        
    Returns:
        DataFrame with accuracy metrics by parameter value
    """
    # Group results by the specified parameter
    param_groups = {}
    
    for item in results:
        param_value = item['input_prompt'].get(param_field, 'unknown')
        if param_value not in param_groups:
            param_groups[param_value] = []
        
        clean_answer = extract_clean_answer(item['answer'])
        question = item['input_prompt'].get('user_query', '').lower().strip()
        
        # Store tuple of (clean_answer, question)
        param_groups[param_value].append((clean_answer, question))
    
    # Calculate metrics for each parameter value
    metrics = []
    
    for param_value, answers_questions in param_groups.items():
        # Extract answers and corresponding questions
        answers, questions = zip(*answers_questions)
        
        # Collect ground truth answers for these questions
        ground_truth = []
        found_count = 0
        
        for q in questions:
            if q in annotations['question_to_annotation']:
                gt = annotations['question_to_annotation'][q]['multiple_choice_answer']
                ground_truth.append(gt)
                found_count += 1
            else:
                ground_truth.append("")  # Placeholder for missing ground truth
        
        # Calculate accuracy only on questions with found ground truth
        if found_count > 0:
            # Filter out pairs where ground truth is empty
            valid_pairs = [(a, gt) for a, gt in zip(answers, ground_truth) if gt]
            if valid_pairs:
                valid_answers, valid_gt = zip(*valid_pairs)
                
                total = len(valid_answers)
                exact_match = sum(1 for a, gt in zip(valid_answers, valid_gt) 
                                  if a.lower().strip() == gt.lower().strip())
                fuzzy_match = sum(1 for a, gt in zip(valid_answers, valid_gt)
                                  if a.lower().strip() in gt.lower().strip() or
                                  gt.lower().strip() in a.lower().strip())
                
                metrics.append({
                    param_field: param_value,
                    'total_questions': total,
                    'matched_questions': found_count,
                    'exact_match': exact_match,
                    'exact_match_accuracy': exact_match / total if total > 0 else 0,
                    'fuzzy_match': fuzzy_match,
                    'fuzzy_match_accuracy': fuzzy_match / total if total > 0 else 0,
                })
    
    return pd.DataFrame(metrics)

def visualize_accuracy_by_param(metrics_df, param_field, output_file):
    """
    Create a visualization of accuracy metrics by parameter values.
    
    Args:
        metrics_df: DataFrame with accuracy metrics
        param_field: Parameter field used for grouping
        output_file: Path to save the visualization
    """
    plt.figure(figsize=(10, 6))
    
    x = range(len(metrics_df))
    width = 0.35
    
    # Plot exact match accuracy
    plt.bar(
        [i - width/2 for i in x],
        metrics_df['exact_match_accuracy'],
        width=width,
        label='Exact Match'
    )
    
    # Plot fuzzy match accuracy
    plt.bar(
        [i + width/2 for i in x],
        metrics_df['fuzzy_match_accuracy'],
        width=width,
        label='Fuzzy Match'
    )
    
    # Configure chart
    plt.xlabel(param_field)
    plt.ylabel('Accuracy')
    plt.title(f'VQA Accuracy by {param_field}')
    plt.xticks(x, metrics_df[param_field])
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add data values on bars
    for i, v in enumerate(metrics_df['exact_match_accuracy']):
        plt.text(i - width/2, v + 0.01, f'{v:.2f}', ha='center')
    
    for i, v in enumerate(metrics_df['fuzzy_match_accuracy']):
        plt.text(i + width/2, v + 0.01, f'{v:.2f}', ha='center')
    
    # Save
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def run_evaluation_workflow(args):
    """
    Run the complete evaluation workflow.
    
    Args:
        args: Command line arguments
    """
    print("Starting VQA evaluation workflow...")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    charts_dir = os.path.join(args.output_dir, 'charts')
    os.makedirs(charts_dir, exist_ok=True)
    data_dir = os.path.join(args.output_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # Step 1: Load experiment results
    if args.within_results:
        print(f"Loading within-subject results from {args.within_results}")
        within_results = load_jsonl_to_list(args.within_results)
        print(f"Loaded {len(within_results)} within-subject results")
    else:
        within_results = []
    
    if args.between_results:
        print(f"Loading between-subject results from {args.between_results}")
        between_results = load_jsonl_to_list(args.between_results)
        print(f"Loaded {len(between_results)} between-subject results")
    else:
        between_results = []
    
    # Step 2: Extract VQA annotations using loadset.py
    print(f"Loading VQA val dataset using loadset...")

    vqa_examples = load_vqa_subset()

    # Build annotations using the loaded examples
    annotations = {
        "annotations": vqa_examples,
        "question_to_annotation": {ex['question'].strip().lower(): ex for ex in vqa_examples},
        "image_question_to_annotation": {f"{ex['image_id']}_{ex['question'].strip().lower()}": ex for ex in vqa_examples}
    }
    annotations_file = os.path.join(data_dir, 'vqa_annotations.json')
    with open(annotations_file, 'w') as f:
        json.dump(annotations, f, indent=2)

    print("Creating question ID mapping")
    mapping_file = os.path.join(data_dir, 'question_id_mapping.json')
    question_id_mapping = {ex['question'].strip().lower(): ex['question_id'] for ex in vqa_examples}
    with open(mapping_file, 'w') as f:
        json.dump(question_id_mapping, f, indent=2)
    
    # Step 3: Calculate accuracy metrics by parameter combinations
    param_fields = ['time_budget', 'priority', 'task_complexity', 'response_detail']
    
    # Within-subject analysis
    if within_results:
        print("Analyzing within-subject results...")
        within_param_metrics = {}
        
        for param in param_fields:
            print(f"Calculating metrics grouped by {param}...")
            metrics_df = calculate_param_metrics(within_results, annotations, param)
            within_param_metrics[param] = metrics_df
            
            # Save metrics
            metrics_file = os.path.join(data_dir, f'within_metrics_{param}.csv')
            metrics_df.to_csv(metrics_file, index=False)
            
            # Create visualization
            chart_file = os.path.join(charts_dir, f'within_accuracy_{param}.png')
            visualize_accuracy_by_param(metrics_df, param, chart_file)
    
    # Between-subject analysis
    if between_results:
        print("Analyzing between-subject results...")
        between_param_metrics = {}
        
        for param in param_fields:
            print(f"Calculating metrics grouped by {param}...")
            metrics_df = calculate_param_metrics(between_results, annotations, param)
            between_param_metrics[param] = metrics_df
            
            # Save metrics
            metrics_file = os.path.join(data_dir, f'between_metrics_{param}.csv')
            metrics_df.to_csv(metrics_file, index=False)
            
            # Create visualization
            chart_file = os.path.join(charts_dir, f'between_accuracy_{param}.png')
            visualize_accuracy_by_param(metrics_df, param, chart_file)
    
    # Step 4: Compare within and between results if both are available
    if within_results and between_results:
        print("Creating comparative analysis...")
        
        for param in param_fields:
            within_df = within_param_metrics[param]
            between_df = between_param_metrics[param]
            
            # Create comparative chart
            plt.figure(figsize=(12, 6))
            
            # Set width of bars
            barWidth = 0.25
            
            # Set positions of bars on X-axis
            param_values = sorted(set(list(within_df[param]) + list(between_df[param])))
            positions = list(range(len(param_values)))
            
            # Create lookup dictionaries for each dataframe
            within_lookup = {row[param]: row for _, row in within_df.iterrows()}
            between_lookup = {row[param]: row for _, row in between_df.iterrows()}
            
            # Create data for plotting
            within_exact = [within_lookup.get(val, {}).get('exact_match_accuracy', 0) for val in param_values]
            within_fuzzy = [within_lookup.get(val, {}).get('fuzzy_match_accuracy', 0) for val in param_values]
            between_exact = [between_lookup.get(val, {}).get('exact_match_accuracy', 0) for val in param_values]
            between_fuzzy = [between_lookup.get(val, {}).get('fuzzy_match_accuracy', 0) for val in param_values]
            
            # Create bars
            plt.bar([p - barWidth for p in positions], within_exact, width=barWidth, label='Within-Subject Exact')
            plt.bar(positions, within_fuzzy, width=barWidth, label='Within-Subject Fuzzy')
            plt.bar([p + barWidth for p in positions], between_exact, width=barWidth, label='Between-Subject Exact')
            plt.bar([p + 2*barWidth for p in positions], between_fuzzy, width=barWidth, label='Between-Subject Fuzzy')
            
            # Configure chart
            plt.xlabel(param)
            plt.ylabel('Accuracy')
            plt.title(f'VQA Accuracy Comparison by {param}')
            plt.xticks([p + barWidth/2 for p in positions], param_values)
            plt.legend()
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Save
            plt.tight_layout()
            chart_file = os.path.join(charts_dir, f'comparison_accuracy_{param}.png')
            plt.savefig(chart_file)
            plt.close()
    
    # Step 5: Generate the best parameter combinations based on accuracy
    if within_results:
        print("Determining best parameter combinations for within-subject design...")
        within_combined_metrics = []
        
        # Group by all parameter combinations
        param_combinations = defaultdict(list)
        
        for item in within_results:
            config_key = (
                item['input_prompt'].get('time_budget', ''),
                item['input_prompt'].get('priority', ''),
                item['input_prompt'].get('task_complexity', ''),
                item['input_prompt'].get('response_detail', '')
            )
            clean_answer = extract_clean_answer(item['answer'])
            question = item['input_prompt'].get('user_query', '').lower().strip()
            param_combinations[config_key].append((clean_answer, question))
        
        # Calculate metrics for each combination
        for config_key, answers_questions in param_combinations.items():
            time_budget, priority, task_complexity, response_detail = config_key
            
            # Extract answers and corresponding questions
            answers, questions = zip(*answers_questions)
            
            # Collect ground truth answers for these questions
            ground_truth = []
            found_count = 0
            
            for q in questions:
                if q in annotations['question_to_annotation']:
                    gt = annotations['question_to_annotation'][q]['multiple_choice_answer']
                    ground_truth.append(gt)
                    found_count += 1
                else:
                    ground_truth.append("")  # Placeholder for missing ground truth
            
            # Calculate accuracy only on questions with found ground truth
            if found_count > 0:
                # Filter out pairs where ground truth is empty
                valid_pairs = [(a, gt) for a, gt in zip(answers, ground_truth) if gt]
                if valid_pairs:
                    valid_answers, valid_gt = zip(*valid_pairs)
                    
                    total = len(valid_answers)
                    exact_match = sum(1 for a, gt in zip(valid_answers, valid_gt) 
                                      if a.lower().strip() == gt.lower().strip())
                    fuzzy_match = sum(1 for a, gt in zip(valid_answers, valid_gt)
                                      if a.lower().strip() in gt.lower().strip() or
                                      gt.lower().strip() in a.lower().strip())
                    
                    within_combined_metrics.append({
                        'time_budget': time_budget,
                        'priority': priority,
                        'task_complexity': task_complexity,
                        'response_detail': response_detail,
                        'total_questions': total,
                        'matched_questions': found_count,
                        'exact_match': exact_match,
                        'exact_match_accuracy': exact_match / total if total > 0 else 0,
                        'fuzzy_match': fuzzy_match,
                        'fuzzy_match_accuracy': fuzzy_match / total if total > 0 else 0,
                    })
        
        # Save combined metrics
        within_combined_df = pd.DataFrame(within_combined_metrics)
        combined_file = os.path.join(data_dir, 'within_combined_metrics.csv')
        within_combined_df.to_csv(combined_file, index=False)
        
        # Find best parameter combinations
        best_exact = within_combined_df.loc[within_combined_df['exact_match_accuracy'].idxmax()]
        best_fuzzy = within_combined_df.loc[within_combined_df['fuzzy_match_accuracy'].idxmax()]
        
        print("\nBest parameter combination for exact match accuracy:")
        print(f"  Time Budget: {best_exact['time_budget']}")
        print(f"  Priority: {best_exact['priority']}")
        print(f"  Task Complexity: {best_exact['task_complexity']}")
        print(f"  Response Detail: {best_exact['response_detail']}")
        print(f"  Accuracy: {best_exact['exact_match_accuracy']:.4f}")
        
        print("\nBest parameter combination for fuzzy match accuracy:")
        print(f"  Time Budget: {best_fuzzy['time_budget']}")
        print(f"  Priority: {best_fuzzy['priority']}")
        print(f"  Task Complexity: {best_fuzzy['task_complexity']}")
        print(f"  Response Detail: {best_fuzzy['response_detail']}")
        print(f"  Accuracy: {best_fuzzy['fuzzy_match_accuracy']:.4f}")
    
    print(f"\nEvaluation workflow complete. Results saved to {args.output_dir}")
    print(f"Charts saved to {charts_dir}")
    print(f"Data files saved to {data_dir}")

def main():
    parser = argparse.ArgumentParser(description='Run VQA evaluation workflow')
    parser.add_argument('--output_dir', default='../evaluation_results', help='Output directory for evaluation results')
    parser.add_argument('--within_results', help='Path to within-subject results file')
    parser.add_argument('--between_results', help='Path to between-subject results file')
    args = parser.parse_args()
    try:
        run_evaluation_workflow(args)
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()