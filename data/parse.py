import json
import argparse

def process_split(image_paths_file, findings_file, impressions_file, output_json_file):
    # Read image paths and findings from the input files
    with open(image_paths_file, 'r') as f:
        image_path_lines = f.read().splitlines()
    
    with open(findings_file, 'r') as f:
        findings = f.read().splitlines()
    
    with open(impressions_file, 'r') as f:
        imps = f.read().splitlines()
    
    # Ensure that the number of image path lines and findings match
    if len(image_path_lines) != len(findings) or len(image_path_lines) != len(imps):
        print("Error: The number of image path lines, findings, and impressions doesn't match.")
        exit(1)
    
    all_entries = []
    
    for i in range(len(image_path_lines)):
        image_paths = image_path_lines[i].split(',')  # Split multiple image paths
        
        entries = [{"image": image_path.strip(), "text": findings[i], "impression": imps[i]} for image_path in image_paths]
        all_entries.extend(entries)
    
    # Write all entries to the single JSON file
    with open(output_json_file, 'w') as json_file:
        json.dump(all_entries, json_file, indent=4)
    
    print(f"Created {output_json_file} with all entries.")


def main():
    parser = argparse.ArgumentParser(description="Process radiology reports into JSON format.")
    parser.add_argument("--image_paths_file", type=str, required=True, help="Path to the image paths file.")
    parser.add_argument("--findings_file", type=str, required=True, help="Path to the findings file.")
    parser.add_argument("--impressions_file", type=str, required=True, help="Path to the impressions file.")
    parser.add_argument("--output_json_file", type=str, required=True, help="Output JSON file path.")
    
    args = parser.parse_args()
    
    process_split(args.image_paths_file, args.findings_file, args.impressions_file, args.output_json_file)

if __name__ == "__main__":
    main()
