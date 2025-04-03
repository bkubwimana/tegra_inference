# VQA Evaluation for Prompt Engineering Experiments

This suite of tools helps you evaluate the performance of prompt engineering experiments on VQA tasks. The tools allow you to:

1. Process experimental results from prompt engineering tests
2. Extract clean answers from model outputs
3. Compare against ground truth VQA annotations
4. Calculate accuracy metrics by prompt parameters
5. Generate visualizations and reports

## Workflow Overview

The complete evaluation workflow consists of:

```
+---------------------+    +------------------------+    +------------------------+
| Experiment Results  | -> | VQA Dataset Processing | -> | Evaluation & Analysis  |
| (JSONL files)       |    | (Annotations)          |    | (Metrics & Charts)     |
+---------------------+    +------------------------+    +------------------------+
```

## Quick Start

1. First, ensure your experiment results are saved in JSONL format.

2. Prepare the VQA annotations:
   ```bash
   python prepare_vqa_annotations.py \
       --vqa_dataset_path /mnt/huggingface_cache/vqa_data \
       --output_file ../data/vqa_annotations.json \
       --mapping_file ../data/question_id_mapping.json
   ```

3. Run the end-to-end evaluation:
   ```bash
   python run_vqa_evaluation.py \
       --within_results ../prompt_within.jsonl \
       --between_results ../prompt_between.jsonl \
       --vqa_dataset_path /mnt/huggingface_cache/vqa_data \
       --output_dir ../evaluation_results
   ```

## Tools Description

### 1. `prepare_vqa_annotations.py`
- Extracts question-answer pairs from VQA dataset
- Creates mappings between questions and their ground truth annotations
- Prepares annotations for evaluation

### 2. `vqa_eval.py`
- Basic analysis functions for VQA results
- Calculates token statistics and creates visualizations
- Handles both within-subject and between-subject experimental designs

### 3. `run_vqa_evaluation.py`
- End-to-end evaluation workflow
- Calculates accuracy metrics by prompt parameters
- Generates visualizations and reports
- Identifies best parameter combinations

## Output Format

The evaluation generates the following outputs:

1. **Data Files**
   - Accuracy metrics by parameter combinations
   - Best parameter combinations
   - Processed annotations

2. **Visualizations**
   - Accuracy comparisons by parameter
   - Token usage by parameter
   - Comparative analysis of within/between-subject designs

## Using with Standard VQA Evaluation Scripts

The `convert_to_vqa_eval_format` function in `output_processor.py` converts your results to the standard VQA evaluation format, which can be used with official VQA evaluation scripts:

```python
from src.output_processor import convert_to_vqa_eval_format

# Convert your results to VQA format
convert_to_vqa_eval_format(
    jsonl_file="path/to/your/results.jsonl",
    output_file="path/to/save/vqa_format.json",
    question_id_mapping=question_id_mapping  # Optional
)
```

## Best Practices

1. **Consistent Formatting**: Ensure your model outputs have consistent formatting with BEGIN_RESPONSE/END_RESPONSE markers.

2. **Question Matching**: For accurate evaluation, ensure your experiment questions match the VQA dataset questions.

3. **Ground Truth**: Always use the official VQA annotations as ground truth for fair comparison.

4. **Parameter Analysis**: Analyze performance across different parameter combinations to identify optimal prompt configurations.

## Troubleshooting

- **Missing Ground Truth**: If many questions don't have matching ground truth, verify question formatting and compare with original VQA questions.
- **Low Accuracy**: Check the `extract_clean_answer` function if the accuracy is unexpectedly low, as it might not be correctly parsing your model's answers.
- **Memory Issues**: For large datasets, process results in smaller batches or use a machine with more memory.

## Further Development

You can extend these tools to:
1. Support more VQA datasets
2. Add more sophisticated analysis metrics
3. Include statistical significance testing
4. Integrate with continuous evaluation pipelines