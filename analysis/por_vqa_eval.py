#!/usr/bin/env python3
"""
End-to-End VQA Evaluation Workflow (Custom VQA v2 Accuracy with Official Normalization)

This script runs a complete evaluation workflow for VQA experiments:
1. Processes prompt engineering experiment results
2. Uses the official VQA v2 normalization functions from the VQAEval module to clean answers
3. Compares against ground truth VQA annotations
4. Calculates VQA v2–style accuracy metrics by prompt parameters
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

# Add parent directory to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from tests.loadset import load_vqa_subset  
from normalize import VQAEval
from vqa_eval import load_jsonl_to_list

###############################################################################
# Create a dummy VQA and VQARes object to instantiate VQAEval for normalization
###############################################################################
class DummyVQA:
    def getQuesIds(self):
        return []  # Not used in our normalization

class DummyVQARes:
    pass

# Instantiate VQAEval solely to access its normalization methods.
dummy_vqa = DummyVQA()
dummy_vqa_res = DummyVQARes()
norm = VQAEval(dummy_vqa, dummy_vqa_res)

###############################################################################
# New normalization helper using the official VQA v2 functions
###############################################################################
def normalize_answer_with_module(answer: str) -> str:
    """
    Normalize an answer string using the VQAEval module's processing.
    """
    # First remove newlines and extra whitespace
    answer = answer.replace('\n', ' ').replace('\t', ' ').strip()
    # Apply punctuation processing then digit/article processing
    answer = norm.processPunctuation(answer)
    answer = norm.processDigitArticle(answer)
    return answer

###############################################################################
# Custom VQA v2 Accuracy Functions using the official normalization
###############################################################################
def calculate_vqa_v2_accuracy(predicted_answer, ground_truth_answers):
    """
    Compute VQA v2 accuracy for a single question using official normalization.
    
    Args:
        predicted_answer: string prediction.
        ground_truth_answers: list of ground truth answers (usually 10 answers).
        
    Returns:
        Accuracy value (float between 0 and 1).
    """
    # Normalize predicted answer using the module's functions
    norm_pred = normalize_answer_with_module(predicted_answer)
    # Normalize each ground truth answer
    norm_gts = [normalize_answer_with_module(ans) for ans in ground_truth_answers]
    count = sum(1 for ans in norm_gts if norm_pred == ans)
    return min(count / 3.0, 1)

def calculate_param_vqa_v2_metrics(results, annotations, param_field):
    """
    Calculate VQA v2 accuracy metrics grouped by a specific prompt parameter.
    
    Args:
        results: List of result dictionaries.
        annotations: Dictionary with VQA annotations.
        param_field: Parameter field to group by (e.g., 'time_budget').
        
    Returns:
        DataFrame with VQA v2 accuracy metrics by parameter value.
    """
    # Group results by the specified parameter
    param_groups = defaultdict(list)
    for item in results:
        param_value = item['input_prompt'].get(param_field, 'unknown')
        param_groups[param_value].append(item)
    
    metrics = []
    for param_value, items in param_groups.items():
        accuracies = []
        for item in items:
            # Normalize the predicted answer using the module's normalization
            norm_answer = normalize_answer_with_module(item['answer'])
            # Use normalized question text as key for lookup
            q_text = item['input_prompt'].get('user_query', '').lower().strip()
            if q_text in annotations['question_to_annotation']:
                gt_entry = annotations['question_to_annotation'][q_text]
                # Expecting the ground truth entry to have an 'answers' field
                if 'answers' in gt_entry:
                    gt_answers = [ans['answer'] if isinstance(ans, dict) else ans 
                                  for ans in gt_entry['answers']]
                else:
                    gt_answers = [gt_entry.get('multiple_choice_answer', '')]
                if gt_answers:
                    acc = calculate_vqa_v2_accuracy(norm_answer, gt_answers)
                    accuracies.append(acc)
        if accuracies:
            group_accuracy = sum(accuracies) / len(accuracies)
            metrics.append({
                param_field: param_value,
                'num_questions': len(accuracies),
                'vqa_v2_accuracy': group_accuracy
            })
    return pd.DataFrame(metrics)

def visualize_accuracy_by_param(metrics_df, param_field, output_file):
    """
    Create a visualization of VQA v2 accuracy metrics by parameter values.
    
    Args:
        metrics_df: DataFrame with VQA v2 accuracy metrics.
        param_field: Parameter field used for grouping.
        output_file: Path to save the visualization.
    """
    plt.figure(figsize=(10, 6))
    x = range(len(metrics_df))
    
    plt.bar(x, metrics_df['vqa_v2_accuracy'], width=0.6, label='VQA v2 Accuracy')
    
    plt.xlabel(param_field)
    plt.ylabel('VQA v2 Accuracy')
    plt.title(f'VQA v2 Accuracy by {param_field}')
    plt.xticks(x, metrics_df[param_field])
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Annotate bars with accuracy values
    for i, v in enumerate(metrics_df['vqa_v2_accuracy']):
        plt.text(i, v + 0.01, f'{v:.2f}', ha='center')
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

###############################################################################
# Evaluation Workflow
###############################################################################
def run_evaluation_workflow(args):
    """
    Run the complete evaluation workflow.
    
    Args:
        args: Command line arguments.
    """
    print("Starting VQA evaluation workflow (Custom VQA v2 with Official Normalization)...")
    
    # Create output directories
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
    
    # Step 2: Load VQA annotations using loadset.py
    print("Loading VQA val dataset using loadset...")
    vqa_examples = load_vqa_subset()
    
    # Build annotations dictionary
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
    
    # Step 3: Calculate VQA v2 accuracy metrics by prompt parameters
    param_fields = ['time_budget', 'priority', 'task_complexity', 'response_detail']
    
    # Within-subject analysis
    if within_results:
        print("Analyzing within-subject results...")
        within_param_metrics = {}
        for param in param_fields:
            print(f"Calculating metrics grouped by {param}...")
            metrics_df = calculate_param_vqa_v2_metrics(within_results, annotations, param)
            within_param_metrics[param] = metrics_df
            
            # Save metrics to CSV
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
            metrics_df = calculate_param_vqa_v2_metrics(between_results, annotations, param)
            between_param_metrics[param] = metrics_df
            
            # Save metrics to CSV
            metrics_file = os.path.join(data_dir, f'between_metrics_{param}.csv')
            metrics_df.to_csv(metrics_file, index=False)
            
            # Create visualization
            chart_file = os.path.join(charts_dir, f'between_accuracy_{param}.png')
            visualize_accuracy_by_param(metrics_df, param, chart_file)
    
    # Step 4: Comparative Analysis (if both within and between results are available)
    if within_results and between_results:
        print("Creating comparative analysis...")
        for param in param_fields:
            within_df = within_param_metrics[param]
            between_df = between_param_metrics[param]
            
            plt.figure(figsize=(12, 6))
            barWidth = 0.3
            # Determine the union of parameter values
            param_values = sorted(set(list(within_df[param]) + list(between_df[param])))
            positions = list(range(len(param_values)))
            
            # Create lookup dictionaries
            within_lookup = {row[param]: row for _, row in within_df.iterrows()}
            between_lookup = {row[param]: row for _, row in between_df.iterrows()}
            
            within_acc = [within_lookup.get(val, {}).get('vqa_v2_accuracy', 0) for val in param_values]
            between_acc = [between_lookup.get(val, {}).get('vqa_v2_accuracy', 0) for val in param_values]
            
            plt.bar([p - barWidth/2 for p in positions], within_acc, width=barWidth, label='Within-Subject')
            plt.bar([p + barWidth/2 for p in positions], between_acc, width=barWidth, label='Between-Subject')
            
            plt.xlabel(param)
            plt.ylabel('VQA v2 Accuracy')
            plt.title(f'VQA v2 Accuracy Comparison by {param}')
            plt.xticks(positions, param_values)
            plt.ylim(0, 1)
            plt.legend()
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            chart_file = os.path.join(charts_dir, f'comparison_accuracy_{param}.png')
            plt.tight_layout()
            plt.savefig(chart_file)
            plt.close()
    
    # Step 5: Determine the Best Parameter Combination (Within-Subject)
    if within_results:
        print("Determining best parameter combination for within-subject design...")
        within_combined_metrics = []
        param_combinations = defaultdict(list)
        
        for item in within_results:
            config_key = (
                item['input_prompt'].get('time_budget', ''),
                item['input_prompt'].get('priority', ''),
                item['input_prompt'].get('task_complexity', ''),
                item['input_prompt'].get('response_detail', '')
            )
            param_combinations[config_key].append(item)
        
        
        for config_key, items in param_combinations.items():
            time_budget, priority, task_complexity, response_detail = config_key
            accuracies = []
            for item in items:
                # Normalize predicted answer
                norm_answer = normalize_answer_with_module(item['answer'])
                q_text = item['input_prompt'].get('user_query', '').lower().strip()
                if q_text in annotations['question_to_annotation']:
                    gt_entry = annotations['question_to_annotation'][q_text]
                    if 'answers' in gt_entry:
                        gt_answers = [ans['answer'] if isinstance(ans, dict) else ans 
                                      for ans in gt_entry['answers']]
                    else:
                        gt_answers = [gt_entry.get('multiple_choice_answer', '')]
                    if gt_answers:
                        acc = calculate_vqa_v2_accuracy(norm_answer, gt_answers)
                        accuracies.append(acc)
            if accuracies:
                group_acc = sum(accuracies) / len(accuracies)
                within_combined_metrics.append({
                    'time_budget': time_budget,
                    'priority': priority,
                    'task_complexity': task_complexity,
                    'response_detail': response_detail,
                    'num_questions': len(accuracies),
                    'vqa_v2_accuracy': group_acc
                })
        
        within_combined_df = pd.DataFrame(within_combined_metrics)
        combined_file = os.path.join(data_dir, 'within_combined_metrics.csv')
        within_combined_df.to_csv(combined_file, index=False)
        
        best_combination = within_combined_df.loc[within_combined_df['vqa_v2_accuracy'].idxmax()]
        
        print("\nBest parameter combination for VQA v2 accuracy:")
        print(f"  Time Budget: {best_combination['time_budget']}")
        print(f"  Priority: {best_combination['priority']}")
        print(f"  Task Complexity: {best_combination['task_complexity']}")
        print(f"  Response Detail: {best_combination['response_detail']}")
        print(f"  VQA v2 Accuracy: {best_combination['vqa_v2_accuracy']:.4f}")
    
    print(f"\nEvaluation workflow complete. Results saved to {args.output_dir}")
    print(f"Charts saved to {charts_dir}")
    print(f"Data files saved to {data_dir}")

###############################################################################
# Main Entry Point
###############################################################################
def main():
    parser = argparse.ArgumentParser(description='Run VQA evaluation workflow (Custom VQA v2 with Official Normalization)')
    parser.add_argument('--output_dir', default='../eval', help='Output directory for evaluation results')
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
