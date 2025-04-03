import json
import sys
import statistics
from collections import Counter, defaultdict
import numpy as np
import os  

def load_json_objects(file_path):
    results = []
    decoder = json.JSONDecoder()
    with open(file_path, "r") as f:
        content = f.read().strip()
    
    pos = 0
    length = len(content)
    while pos < length:
        try:
            obj, index = decoder.raw_decode(content, pos)
            results.append(obj)
            pos = index
            while pos < length and content[pos].isspace():
                pos += 1
        except Exception as e:
            print(f"Error parsing JSON at position {pos}: {e}")
            break
    return results

def analyze_results(results):
    total_latencies = []
    generation_latencies = []
    decode_latencies = []
    token_counts = []
    time_budgets = []
    priorities = []
    task_complexities = []
    prompt_validity_count = 0

    for obj in results:
        if "latency" in obj:
            latency = obj["latency"]
            if "total" in latency:
                total_latencies.append(latency["total"])
            if "generation" in latency:
                generation_latencies.append(latency["generation"])
            if "decode" in latency:
                decode_latencies.append(latency["decode"])
        if "token_count" in obj:
            token_counts.append(obj["token_count"])
        if "input_prompt" in obj:
            ip = obj["input_prompt"]
            if "time_budget" in ip:
                time_budgets.append(ip["time_budget"])
            if "priority" in ip:
                priorities.append(ip["priority"])
            if "task_complexity" in ip:
                task_complexities.append(ip["task_complexity"])
        answer = obj.get("answer", "")
        if "BEGIN_RESPONSE" in answer and "END_RESPONSE" in answer:
            prompt_validity_count += 1

    analysis = {}
    analysis["num_results"] = len(results)
    if total_latencies:
        analysis["total_latency_avg"] = statistics.mean(total_latencies)
        analysis["total_latency_std"] = statistics.stdev(total_latencies) if len(total_latencies) > 1 else 0
    if generation_latencies:
        analysis["generation_latency_avg"] = statistics.mean(generation_latencies)
    if decode_latencies:
        analysis["decode_latency_avg"] = statistics.mean(decode_latencies)
    if token_counts:
        analysis["avg_token_count"] = statistics.mean(token_counts)
        # analysis["min_token_count"] = min(token_counts)
        analysis["max_token_count"] = max(token_counts)
        # New change: count responses with token_count above 120
        high_tokens = [t for t in token_counts if t > 120]
        for t in high_tokens:
            print(f"High token count: {t} (index: {token_counts.index(t)})")
        analysis["high_token_count"] = len(high_tokens)
    analysis["time_budget_freq"] = dict(Counter(time_budgets))
    analysis["priority_freq"] = dict(Counter(priorities))
    analysis["task_complexity_freq"] = dict(Counter(task_complexities))
    analysis["response_format_complete"] = f"{prompt_validity_count} out of {len(results)} responses contained BEGIN_RESPONSE and END_RESPONSE"
    return analysis

def plot_token_ranges(results):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed. Please install it to use the plotting feature.")
        sys.exit(1)
        
    expected_mins = []
    expected_maxs = []
    actual_token_counts = []
    indices = []
    
    for idx, obj in enumerate(results):
        ip = obj.get("input_prompt", {})
        if "max_tokens" in ip and "token_count" in obj:
            try:
                min_val = int(ip["min_tokens"]) if "min_tokens" in ip else 0
                max_val = int(ip["max_tokens"])
                token_count = obj["token_count"]
            except ValueError:
                continue
            expected_mins.append(min_val)
            expected_maxs.append(max_val)
            actual_token_counts.append(token_count)
            indices.append(idx)
    
    if not indices:
        print("No valid token range data found for plotting.")
        return
    
    lower_errors = [abs(act - mini) for act, mini in zip(actual_token_counts, expected_mins)]
    upper_errors = [abs(maxi - act) for act, maxi in zip(actual_token_counts, expected_maxs)]
    
    plt.figure(figsize=(10,6))
    plt.errorbar(
        indices, 
        actual_token_counts, 
        yerr=[lower_errors, upper_errors], 
        fmt='o', 
        ecolor='red', 
        capsize=5, 
        label="Token Count"
    )
    plt.xlabel("Result Index")
    plt.ylabel("Token Count")
    plt.title("Actual Token Count vs. Expected Boundaries")
    plt.legend()
    plt.tight_layout()
    plt.savefig("token_ranges.png")

def plot_tokens_by_category(results, category_key):
    """
    Group actual token counts by a given category (e.g., "priority", "task_complexity", or "time_budget")
    and overlay reference lines for the average expected min and max for that group.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed. Please install it to use the plotting feature.")
        sys.exit(1)
        
    from collections import defaultdict

    grouped_tokens = defaultdict(list)
    grouped_bounds = defaultdict(list)
    
    for obj in results:
        ip = obj.get("input_prompt", {})
        if category_key in ip and "token_count" in obj and "max_tokens" in ip:
            group = ip[category_key]
            token_count = obj["token_count"]
            try:
                min_val = int(ip["min_tokens"]) if "min_tokens" in ip else 0
                max_val = int(ip["max_tokens"])
            except ValueError:
                continue
            grouped_tokens[group].append(token_count)
            grouped_bounds[group].append((min_val, max_val))
    
    if not grouped_tokens:
        print(f"No valid data found for category '{category_key}'.")
        return
    
    groups = list(grouped_tokens.keys())
    data = [grouped_tokens[g] for g in groups]

    plt.figure(figsize=(10,6))
    box = plt.boxplot(data, labels=groups, patch_artist=True)
    
    # Instead of computing a y coordinate from the data,
    # use the x-axis transformation to place text below the x-axis labels.
    ax = plt.gca()  # current axis
    for i, group in enumerate(groups):
        bounds = grouped_bounds[group]
        avg_min = statistics.mean([b[0] for b in bounds])
        avg_max = statistics.mean([b[1] for b in bounds])
        # plt.hlines(avg_min, i+0.8, i+1.2, colors='green', linestyles='dashed',
        #            label="Expected Min" if i == 0 else "")
        plt.hlines(avg_max, i+0.8, i+1.2, colors='blue', linestyles='dashed',
                   label="Expected Max" if i == 0 else "")
            
        tokens = np.array(grouped_tokens[group])
        outliers = tokens[(tokens > avg_max)]
        count_outliers = len(outliers)
        total_tokens = len(tokens)
        percent_outliers = (count_outliers / total_tokens) * 100 if total_tokens > 0 else 0
            
        annotation = f"Maxliers: {count_outliers}/{total_tokens} ({percent_outliers:.1f}%)"
        # Place the annotation below the x-axis (using the x-axis transform)
        ax.text(i+1, -0.1, annotation, transform=ax.get_xaxis_transform(), 
                horizontalalignment='center', verticalalignment='top',
                color='red', fontsize=10)
        
    plt.xlabel(category_key.capitalize())
    plt.ylabel("Token Count")
    plt.title(f"Token Counts by {category_key.capitalize()}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"token_counts_by_{category_key}.png")

def plot_scatter_reference(results):
    """
    Create a scatter plot showing the actual token counts and reference points for the expected minimum
    and maximum for each result.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed. Please install it to use the plotting feature.")
        sys.exit(1)
        
    indices = []
    actual_tokens = []
    expected_min_list = []
    expected_max_list = []
    
    for idx, obj in enumerate(results):
        ip = obj.get("input_prompt", {})
        if "max_tokens" in ip and "token_count" in obj:
            try:
                min_val = int(ip["min_tokens"]) if "min_tokens" in ip else 0
                max_val = int(ip["max_tokens"])
            except ValueError:
                continue
            indices.append(idx)
            actual_tokens.append(obj["token_count"])
            expected_min_list.append(min_val)
            expected_max_list.append(max_val)
    
    if not indices:
        print("No valid data found for scatter plot.")
        return
    
    plt.figure(figsize=(10,6))
    # Scatter plot for actual token counts
    plt.scatter(indices, actual_tokens, color='black', label="Actual Token Count")
    # Scatter plot for expected minimums
    # Scatter plot for expected maximums
    plt.scatter(indices, expected_max_list, color='blue', marker='x', label="Expected Max")
    
    plt.xlabel("Result Index")
    plt.ylabel("Token Count")
    plt.title("Actual vs Expected Token Boundaries")
    plt.legend()
    plt.tight_layout()
    plt.savefig("scatter_token_reference.png")

def load_free_results(file_path):
    # Load free test results using normal JSON load (since file is a complete JSON object)
    with open(file_path, "r") as f:
        data = json.load(f)
    return data

def plot_free_results(free_data):
    # Extract token counts and decode latencies from free results
    token_counts = [r["token_count"] for r in free_data["results"]]
    # Some results have 'decode' latency
    latencies = [r["latency"]["decode"] for r in free_data["results"] if "decode" in r["latency"]]
    
    # Plot using matplotlib
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12,5))
    
    plt.subplot(1,2,1)
    plt.hist(token_counts, bins=20, color='skyblue')
    plt.title("Token Count Distribution (Free Test)")
    plt.xlabel("Token Count")
    plt.ylabel("Frequency")
    
    plt.subplot(1,2,2)
    plt.hist(latencies, bins=20, color='lightgreen')
    plt.title("Latency Distribution (Free Test)")
    plt.xlabel("Latency (s)")
    plt.ylabel("Frequency")
    
    plt.tight_layout()
    plt.savefig("reference.png")
    
def plot_free_with_prompt(prompt_results, free_data):
    """
    Plot a combined histogram of token counts from free test data and prompt-engineered data,
    along with a second graph for latency distributions with overlay.
    """
    import matplotlib.pyplot as plt
    
    # Extract token counts
    free_tokens = [r["token_count"] for r in free_data["results"] if "token_count" in r]
    prompt_tokens = [obj["token_count"] for obj in prompt_results if "token_count" in obj]
    
    # Extract decode latencies
    free_latencies = [r["latency"]["decode"] for r in free_data["results"]
                      if "latency" in r and "decode" in r["latency"]]
    prompt_latencies = [obj["latency"]["decode"] for obj in prompt_results
                        if "latency" in obj and "decode" in obj["latency"]]
    
    # Create figure with 2 subplots
    plt.figure(figsize=(12,6))
    
    # Subplot for token counts
    plt.subplot(1,2,1)
    plt.hist(free_tokens, bins=20, color='skyblue', alpha=0.7, label="Free Test")
    plt.hist(prompt_tokens, bins=20, color='coral', alpha=0.5, label="Prompt-engineered")
    plt.xlabel("Token Count")
    plt.ylabel("Frequency")
    plt.title("Token Count Distributions: Free Test vs. Prompt-engineered")
    plt.legend()
    
    # Subplot for latencies
    plt.subplot(1,2,2)
    plt.hist(free_latencies, bins=20, color='skyblue', alpha=0.7, label="Free Test Latency")
    plt.hist(prompt_latencies, bins=20, color='coral', alpha=0.5, label="Prompt-engineered Latency")
    plt.xlabel("Latency (s)")
    plt.ylabel("Frequency")
    plt.title("Latency Distribution: Free Test vs. Prompt-engineered")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("free_with_prompt.png")
    
def plot_combined_scatter(prompt_results, free_data):
    """
    Create a scatter plot showing token counts for each prompt-engineered result
    and overlay the free test token counts on the same figure for a query-by-query comparison.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed.")
        sys.exit(1)
    
    # Extract token counts for prompt-engineered results
    prompt_indices = list(range(len(prompt_results)))
    prompt_tokens = [obj["token_count"] for obj in prompt_results if "token_count" in obj]

    # Extract token counts for free test results
    free_results = free_data.get("results", [])
    free_indices = list(range(len(free_results)))
    free_tokens = [obj["token_count"] for obj in free_results if "token_count" in obj]

    plt.figure(figsize=(10,6))
    plt.scatter(prompt_indices, prompt_tokens, color='black', label="Prompt-engineered")
    plt.scatter(free_indices, free_tokens, marker='x', color='red', label="Free Test")
    
    plt.xlabel("Result Index")
    plt.ylabel("Token Count")
    plt.title("Query-wise Token Counts Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig("combined_token_scatter.png")

def analyze_free_results(free_data):
    """
    Analyze free test results similar to prompt-engineered results.
    """
    results = free_data.get("results", [])
    return analyze_results(results)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analysis script for deepseek results")
    parser.add_argument("file", type=str, help="Path to prompt_results.txt")
    parser.add_argument("--ref", type=str, help="Path to free_results.txt", default="/mnt/packages/models/deepseek/janus/results/free_results.json")
    parser.add_argument("--plot", action="store_true", help="Plot prompt-engineered token ranges")
    parser.add_argument("--plot-category", type=str, help="Plot tokens by category")
    parser.add_argument("--plot-scatter", action="store_true", help="Plot scatter reference")
    parser.add_argument("--comb", action="store_true", help="Plot combined token scatter")
    parser.add_argument("--overlay", action="store_true", help="Plot free test data overlaid with prompt-engineered data")
    args = parser.parse_args()
    
    free_data = None
    if args.ref and os.path.exists(args.ref):
        free_data = load_free_results(args.ref)
    else:
        print(f"Free results file '{args.ref}' not found. Free test plots will be skipped.")
    
    prompt_data = load_json_objects(args.file)
    if not prompt_data:
        print("No JSON objects were loaded.")
        sys.exit(1)
    
    analysis = analyze_results(prompt_data)
    print("Analysis Summary (Prompt-engineered):")
    print("-------------------------------------")
    for key, value in analysis.items():
        print(f"{key}: {value}")

    if args.plot:
        plot_token_ranges(prompt_data)
    if args.plot_category:
        plot_tokens_by_category(prompt_data, args.plot_category)
    if args.plot_scatter:
        plot_scatter_reference(prompt_data)
    if args.comb and free_data is not None:
        plot_combined_scatter(prompt_data, free_data)
    if args.overlay and free_data is not None:
        plot_free_with_prompt(prompt_data, free_data)