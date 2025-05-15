"""
Comparative analysis module to load and compare inference CSV results
from 'time_budget' (baseline) and 'baseline' (controlled) prompting strategies.
Computes comparative metrics and visualizes data globally.
"""
import argparse
import os
import glob
from pathlib import Path
from typing import Tuple, Optional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class ComparativeDataProcessor:
    """
    Loads and processes CSV result files for comparative inference studies.
    """

    def __init__(self, data_dir: str) -> None:
        """
        Args:
            data_dir: Directory containing result CSV files.
        """
        self.data_dir = data_dir
        self.df_time_budget: Optional[pd.DataFrame] = None
        self.df_baseline: Optional[pd.DataFrame] = None

    def _load_and_filter_data(self, file_pattern: str) -> Optional[pd.DataFrame]:
        """Helper to load and concatenate data for a given file pattern."""
        paths = glob.glob(os.path.join(self.data_dir, file_pattern))
        if not paths:
            print(f"Warning: No files found for pattern '{file_pattern}' in '{self.data_dir}'.")
            return None
        
        frames = [pd.read_csv(p) for p in paths]
        if not frames:
            return None
            
        df_all = pd.concat(frames, ignore_index=True)
        
        if df_all.empty:
            print(f"Warning: No data found for pattern '{file_pattern}'.")
            return None
            
        df_all['correct'] = df_all['predicted_index'] == df_all['ground_truth_index']
        return df_all

    def load_data(self) -> None:
        """
        Load data for time-budget (baseline) and free-run (controlled) experiments.
        Patterns are designed to capture all files ending with baseline.csv or controlled.csv.
        """
        print("Loading data...")
        self.df_time_budget = self._load_and_filter_data("controlled.csv")
        self.df_baseline = self._load_and_filter_data("baseline.csv")

        if self.df_time_budget is None:
            print("Failed to load or filter data for time_budget runs.")
        else:
            print(f"Loaded {len(self.df_time_budget)} records for time_budget runs.")

        if self.df_baseline is None:
            print("Failed to load or filter data for baseline runs.")
        else:
            print(f"Loaded {len(self.df_baseline)} records for baseline runs.")


    def get_summary_statistics(self) -> Optional[pd.DataFrame]:
        """
        Compute summary statistics (accuracy, avg_time, avg_tokens) for both datasets.
        """
        summaries = []
        if self.df_time_budget is not None and not self.df_time_budget.empty:
            summary_tb = pd.Series({
                'run_type': 'Time Budget',
                'accuracy': self.df_time_budget['correct'].mean(),
                'avg_inference_time': self.df_time_budget['inference_time'].mean(),
                'avg_output_tokens': self.df_time_budget['output_tokens'].mean(),
                'num_samples': len(self.df_time_budget)
            })
            summaries.append(summary_tb)
        
        if self.df_baseline is not None and not self.df_baseline.empty:
            summary_fr = pd.Series({
                'run_type': 'Baseline',  # Corrected from 'baseline'
                'accuracy': self.df_baseline['correct'].mean(),
                'avg_inference_time': self.df_baseline['inference_time'].mean(),
                'avg_output_tokens': self.df_baseline['output_tokens'].mean(),
                'num_samples': len(self.df_baseline)
            })
            summaries.append(summary_fr)

        if not summaries:
            return None
        
        return pd.DataFrame(summaries)


class ComparativeDataVisualizer:
    """
    Visualization tools for comparative analysis of inference data.
    """

    @staticmethod
    def plot_metric_comparison_bar(summary_df: pd.DataFrame, metric: str, title: str, ylabel: str, output_dir: str) -> None:
        """
        Plot a bar chart comparing a given metric between run types.
        """
        if summary_df is None or summary_df.empty or metric not in summary_df.columns:
            print(f"Skipping plot '{title}': Data is missing or metric '{metric}' not found.")
            return

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(x='run_type', y=metric, data=summary_df, ax=ax, palette=['skyblue', 'lightcoral'], hue='run_type', legend=False)
        ax.set_title(title)
        ax.set_xlabel('Run Type')
        ax.set_ylabel(ylabel)  # Restored y-axis label
        fig.tight_layout()
        file_path = os.path.join(output_dir, f"{metric.replace(' ', '_').lower()}_comparison.png")
        fig.savefig(file_path, dpi=300)
        plt.close(fig)
        print(f"Saved {metric} comparison plot to {file_path}")

    @staticmethod
    def plot_distribution_comparison(df_time_budget: Optional[pd.DataFrame], df_baseline: Optional[pd.DataFrame], column: str, title: str, xlabel: str, output_dir: str) -> None:
        """
        Plot overlapping histograms for a given column to compare distributions by count.
        """
        if (df_time_budget is None or df_time_budget.empty or column not in df_time_budget.columns) and \
           (df_baseline is None or df_baseline.empty or column not in df_baseline.columns):
            print(f"Skipping plot '{title}': Data is missing for column '{column}'.")
            return

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 6))

        if df_time_budget is not None and not df_time_budget.empty and column in df_time_budget.columns:
            sns.histplot(df_time_budget[column], ax=ax, label='Time Budget', alpha=0.7, element="step", stat="count", kde=False)
        if df_baseline is not None and not df_baseline.empty and column in df_baseline.columns:
            sns.histplot(df_baseline[column], ax=ax, label='Baseline', alpha=0.7, element="step", stat="count", kde=False, color=sns.color_palette()[1])  # Use a different color
        
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Count')  # Changed from 'Density' to 'Count' and restored
        ax.legend()
        fig.tight_layout()
        file_path = os.path.join(output_dir, f"{column.lower()}_distribution_comparison.png")
        fig.savefig(file_path, dpi=300)
        plt.close(fig)
        print(f"Saved {column} distribution comparison plot to {file_path}")


def main(args):
    """
    Entry point to load data, compute comparative metrics, and visualize results.
    """
    processor = ComparativeDataProcessor(args.data_dir)
    

    processor.load_data()

    summary_stats = processor.get_summary_statistics()

    if summary_stats is None or summary_stats.empty:
        print("No summary statistics could be generated. Exiting.")
        return

    print("\nComparative Summary Statistics:")
    print(summary_stats.to_string())

    output_charts_dir = os.path.join(args.data_dir, "comparative_charts")

    ComparativeDataVisualizer.plot_metric_comparison_bar(
        summary_stats, 'accuracy', 'Accuracy Comparison', 'Accuracy', output_charts_dir
    )
    ComparativeDataVisualizer.plot_metric_comparison_bar(
        summary_stats, 'avg_inference_time', 'Average Inference Time Comparison', 'Avg. Time (s)', output_charts_dir
    )
    ComparativeDataVisualizer.plot_metric_comparison_bar(
        summary_stats, 'avg_output_tokens', 'Average Output Tokens Comparison', 'Avg. Tokens', output_charts_dir
    )

    ComparativeDataVisualizer.plot_distribution_comparison(
        processor.df_time_budget, processor.df_baseline, 'inference_time', 
        'Inference Time Distribution Comparison', 'Inference Time (s)', output_charts_dir
    )
    ComparativeDataVisualizer.plot_distribution_comparison(
        processor.df_time_budget, processor.df_baseline, 'output_tokens', 
        'Output Tokens Distribution Comparison', 'Output Tokens', output_charts_dir
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compare MMLU results from 'time_budget' and 'baseline' prompting strategies.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./outputs/reasoning/",
        help="Directory containing the result CSV files."
    )
    # Removed --num_questions argument
    
    # usage:
    # python comparative_analyzer.py --data_dir ./outputs/reasoning/

    cli_args = parser.parse_args()
    main(cli_args)
