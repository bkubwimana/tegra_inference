"""
A script to go through the prompt results json file and count within-subject and between-subject queries
- Within-subject: Same question repeated with different prompt configurations
- Between-subject: Different questions, each possibly repeated with different configurations
"""

import json
import re
import sys
from collections import defaultdict

def count_vqa_results(file_path):
    try:
        with open(file_path, 'r') as f:
            data_array = json.load(f)
            
        print(f"Loaded JSON data with {len(data_array)} entries.")
        
        # Extract images/questions from the data
        query_by_time = defaultdict(list)
        all_time_budgets = set()
        
        # First pass - collect queries and configurations
        for data in data_array:
            input_prompt = data.get("input_prompt", {})
            time_budget = input_prompt.get("time_budget", "unknown")
            all_time_budgets.add(time_budget)
            
            # Try to extract user query from answer text
            answer = data.get("answer", "")
            extracted_query = None
            
            # Look for user query in prompt or extract from answer
            query = input_prompt.get("user_query", "").strip()
            
            if not query and answer:
                # Try different regex patterns to extract the query
                patterns = [
                    r"User Query: ([^\n]+)",
                    r"USER_QUERY: ([^\n]+)",
                    r"\[([^]]+)\]"  # Extract content within first brackets
                ]
                
                for pattern in patterns:
                    match = re.search(pattern, answer)
                    if match:
                        extracted_query = match.group(1).strip()
                        break
                
                if extracted_query:
                    query = extracted_query
            
            if query:
                query_by_time[time_budget].append(query)
        
        # Find within-subject queries (same query repeated across time budgets)
        all_queries = set()
        for queries in query_by_time.values():
            all_queries.update(queries)
        
        within_subject_queries = defaultdict(set)
        between_subject_queries = set()
        
        for query in all_queries:
            # Count in how many different time budgets this query appears
            appearances = sum(1 for time_budget, queries in query_by_time.items() if query in queries)
            
            if appearances > 1:  # If the query appears in multiple time budgets, it's within-subject
                for time_budget in query_by_time.keys():
                    if query in query_by_time[time_budget]:
                        within_subject_queries[query].add(time_budget)
            else:
                between_subject_queries.add(query)
        
        # Print results
        print(f"Total unique queries found: {len(all_queries)}")
        print(f"Within-subject queries (repeated with different time budgets): {len(within_subject_queries)}")
        print(f"Between-subject queries (not repeated): {len(between_subject_queries)}")
        
        print("\nWithin-subject queries:")
        for i, (query, time_budgets) in enumerate(sorted(within_subject_queries.items())):
            print(f"{i+1}. \"{query}\" - appeared with time budgets: {', '.join(sorted(time_budgets))}")
        
        print("\nBetween-subject queries:")
        for i, query in enumerate(sorted(between_subject_queries)):
            print(f"{i+1}. \"{query}\"")

        # Additional stats
        print("\nTime budget distribution:")
        for time_budget in sorted(all_time_budgets):
            print(f"- {time_budget}: {len(query_by_time[time_budget])} queries")

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python count_vqa.py <prompt_results.hubrid.json>")
        sys.exit(1)
    file_path = sys.argv[1]
    count_vqa_results(file_path)

