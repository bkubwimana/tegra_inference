"""
Post-processing module to load inference CSV results, compute metrics, and visualize data.
"""
import os
from typing import Optional
from pathlib import Path
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class DataProcessor:
    """
    Loads and processes CSV result files for inference studies.
    """

    def __init__(self, data_dir: str, subset_name: str = 'electrical_engineering') -> None:
        """
        Args:
            data_dir: Directory containing result CSV files.
            subset_name: Name of the subset to filter data for.
        """
        self.data_dir = data_dir
        self.subset = subset_name
        self.df: Optional[pd.DataFrame] = None

    def load_data(self, pattern: str = "*.csv") -> pd.DataFrame:
        """
        Load all CSV files matching the glob pattern into a single DataFrame.

        Args:
            pattern: Glob pattern for CSV filenames.

        Returns:
            Combined DataFrame of all results.
        """
        paths = glob.glob(os.path.join(self.data_dir, pattern))
        controlled_path = os.path.join(self.data_dir, 'controlled.csv')
        # frames = [pd.read_csv(p) for p in paths]
        if not paths:
            frames = [pd.read_csv(controlled_path)]
        else:
            frames = [pd.read_csv(p) for p in paths]
            
        df_all = pd.concat(frames, ignore_index=True)
        # filter for single subset
        self.df = df_all[df_all['subset'] == self.subset].reset_index(drop=True)
        # mark correct predictions for easy aggregation
        self.df['correct'] = self.df['predicted_index'] == self.df['ground_truth_index']
        return self.df

    def compute_accuracy(self) -> float:
        """
        Compute overall accuracy based on ground_truth_index and predicted_index.

        Returns:
            Accuracy as a fraction between 0 and 1.
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        correct = self.df['ground_truth_index'] == self.df['predicted_index']
        return correct.mean()

    def summary_by_subset(self) -> pd.DataFrame:
        """
        Generate accuracy and timing summary grouped by subset.

        Returns:
            DataFrame with subset, accuracy, avg_inference_time, avg_output_tokens.
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        # Compute metrics per subset using named aggregations to avoid DeprecationWarning
        summary = (
            self.df.groupby('subset', as_index=False)
            .agg(
                accuracy=('correct', 'mean'),
                avg_time=('inference_time', 'mean'),
                avg_tokens=('output_tokens', 'mean')
            )
        )
        return summary

    def token_statistics(self) -> pd.Series:
        """
        Compute descriptive statistics for the output_tokens field.
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        return self.df['output_tokens'].describe()

    def time_token_correlation(self) -> float:
        """
        Compute Pearson correlation between inference_time and output_tokens.
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        return float(self.df['inference_time'].corr(self.df['output_tokens']))


class DataVisualizer:
    """
    Visualization tools for processed inference data.
    """

    @staticmethod
    def plot_accuracy_histogram(processor: DataProcessor, output_dir: str = 'output/charts') -> None:
        """
        Plot histogram of accuracy across subsets and save to file.
        """
        summary = processor.summary_by_subset()
        # ensure output directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots()
        sns.barplot(data=summary, x='subset', y='accuracy', ax=ax)
        ax.set_title('Accuracy by Subset')
        # rotate x labels using tick_params to avoid warning
        ax.tick_params(axis='x', rotation=45)
        fig.tight_layout()
        file_path = os.path.join(output_dir, 'accuracy_by_subset.png')
        fig.savefig(file_path, dpi=300)
        plt.close(fig)
        print(f"Saved accuracy histogram to {file_path}")

    @staticmethod
    def plot_time_distribution(processor: DataProcessor, output_dir: str = 'output/charts') -> None:
        """
        Plot distribution of inference times and save to file.
        """
        if processor.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots()
        sns.histplot(processor.df['inference_time'], kde=True, ax=ax)
        ax.set_xlabel('Inference Time (s)')
        ax.set_title('Inference Time Distribution')
        fig.tight_layout()
        file_path = os.path.join(output_dir, 'inference_time_distribution.png')
        fig.savefig(file_path, dpi=300)
        plt.close(fig)
        print(f"Saved inference time distribution to {file_path}")

    @staticmethod
    def plot_token_distribution(processor: DataProcessor, output_dir: str = 'output/charts') -> None:
        """
        Plot distribution of output token counts and save to file.
        """
        if processor.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots()
        sns.histplot(processor.df['output_tokens'], kde=True, ax=ax)
        ax.set_xlabel('Output Tokens')
        ax.set_title('Output Token Distribution')
        fig.tight_layout()
        file_path = os.path.join(output_dir, 'output_token_distribution.png')
        fig.savefig(file_path, dpi=300)
        plt.close(fig)
        print(f"Saved output token distribution to {file_path}")

    @staticmethod
    def plot_time_vs_tokens(processor: DataProcessor, output_dir: str = 'output/charts') -> None:
        """
        Plot scatter of inference time vs output tokens with regression line and save to file.
        """
        if processor.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots()
        sns.scatterplot(x='output_tokens', y='inference_time', data=processor.df, ax=ax)
        sns.regplot(x='output_tokens', y='inference_time', data=processor.df, scatter=False, ax=ax, color='red')
        ax.set_xlabel('Output Tokens')
        ax.set_ylabel('Inference Time (s)')
        ax.set_title('Inference Time vs Output Tokens')
        fig.tight_layout()
        file_path = os.path.join(output_dir, 'time_vs_tokens.png')
        fig.savefig(file_path, dpi=300)
        plt.close(fig)
        print(f"Saved time vs tokens plot to {file_path}")


def main(data_folder: str) -> None:
    """
    Entry point to load data, compute metrics, and visualize results.

    Args:
        data_folder: Path to the directory containing result CSVs.
    """
    processor = DataProcessor(data_folder)
    df = processor.load_data()
    acc = processor.compute_accuracy()
    print(f"Overall accuracy: {acc:.2%}")

    # show summaries and plots
    # token statistics and correlation
    token_stats = processor.token_statistics()
    print("Output tokens statistics:", token_stats.to_dict())
    corr = processor.time_token_correlation()
    print(f"Inference time vs tokens correlation: {corr:.3f}")
    DataVisualizer.plot_accuracy_histogram(processor, data_folder)
    DataVisualizer.plot_time_distribution(processor, data_folder)
    DataVisualizer.plot_token_distribution(processor, data_folder)
    DataVisualizer.plot_time_vs_tokens(processor, data_folder)


if __name__ == '__main__':  # pragma: no cover
    import sys

    folder = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    main(folder)
