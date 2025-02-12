import json
import argparse
import numpy as np
from tqdm import tqdm
from radgraph import RadGraph
from f1chexbert import F1CheXbert

def process_labels(input_path, output_path, device="cuda"):
    # Load data
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    radgraph = RadGraph()
    chexbert = F1CheXbert(device=device)
    
    # Defining classes
    target_names = [
        "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema",
        "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion", "Pleural Other",
        "Fracture", "Support Devices", "No Finding"
    ]
    target_names_5 = ["Cardiomegaly", "Edema", "Consolidation", "Atelectasis", "Pleural Effusion"]
    target_names_5_index = np.where(np.isin(target_names, target_names_5))[0]
    
    labeled_data = []
    for instance in tqdm(data, desc="Processing reports"):
        report = instance["finding"]
        annotations = radgraph([report])
        report["entities"] = annotations["0"]["entities"]
        label = chexbert.get_label(report)
        report["label"] = (np.array(label)[target_names_5_index]).tolist()
        labeled_data.append(report)
    
    # Save labeled data
    with open(output_path, 'w') as json_file:
        json.dump(labeled_data, json_file, indent=4)
    
    print(f"Labeled data saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Annotate radiology reports with RadGraph and CheXbert labels.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to input JSON file.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save labeled JSON file.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for CheXbert inference (default: cuda).")
    
    args = parser.parse_args()
    
    process_labels(args.input_path, args.output_path, args.device)

if __name__ == "__main__":
    main()
