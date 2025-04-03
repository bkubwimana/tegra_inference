import json
import sys
import re

def convert_to_json_array(input_file, output_file):
    """
    Converts a file containing concatenated JSON objects to a JSON array.
    """
    data = []
    
    # Read the entire file content
    with open(input_file, 'r') as infile:
        content = infile.read()
    
    # Use regex to find all JSON objects
    pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    json_objects = re.findall(pattern, content)
    
    print(f"Found {len(json_objects)} JSON objects")
    
    # Parse each JSON object
    for json_str in json_objects:
        try:
            json_object = json.loads(json_str)
            data.append(json_object)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            print(f"Offending object: {json_str[:100]}...")  # Show only first 100 chars
            # Continue with next object instead of returning False
    
    print(f"Successfully parsed {len(data)} JSON objects")
    
    # Write the array to the output file
    with open(output_file, 'w') as outfile:
        json.dump(data, outfile, indent=2)
    
    return len(data) > 0  # Return True if we processed at least one object

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_json.py <input_file> <output_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    if convert_to_json_array(input_file, output_file):
        print(f"Successfully converted {input_file} to {output_file}")
    else:
        print("Conversion failed.")
