import argparse
import sys
from typing import List
from plapt import Plapt
import json 
import csv
import warnings
warnings.filterwarnings("ignore")

def read_file(file_path: str) -> List[str]:
    with open(file_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def parse_arguments():
    parser = argparse.ArgumentParser(description="PLAPT: Protein-Ligand Affinity Prediction Tool")
    parser.add_argument('-p', '--proteins', nargs='+', help='Protein sequence(s) or path to a file containing protein sequences')
    parser.add_argument('-m', '--molecules', nargs='+', required=True, help='SMILES string(s) or path to a file containing SMILES strings')
    parser.add_argument('-b', '--batch-size', type=int, default=4, help='Batch size for predictions (default: 4)')
    parser.add_argument('-o', '--output', type=str, default='stdout', help='Output file path (default: stdout)')
    return parser.parse_args()

def process_input(input_data: List[str]) -> List[str]:
    if len(input_data) == 1 and input_data[0].endswith('.txt'):
        return read_file(input_data[0])
    return input_data

def write_output(results: List[dict], output_path: str):
    if output_path == 'stdout':
        for result in results:
            print(f"neg_log10_affinity_M: {result['neg_log10_affinity_M']:.4f}, affinity_uM: {result['affinity_uM']:.4f}")
    else:
        if output_path.endswith('.json'):
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=4)
        elif output_path.endswith('.csv'):
            with open(output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['neg_log10_affinity_M', 'affinity_uM'])  # header row
                for result in results:
                    writer.writerow([result['neg_log10_affinity_M'], result['affinity_uM']])
        else:
            with open(output_path, 'w') as f:
                for result in results:
                    f.write(f"neg_log10_affinity_M: {result['neg_log10_affinity_M']:.4f}, affinity_uM: {result['affinity_uM']:.4f}\n")

def main():
    args = parse_arguments()
    
    molecules = process_input(args.molecules)
    
    plapt = Plapt()
    
    if args.proteins:
        proteins = process_input(args.proteins)
        
        if len(proteins) == 1:
            results = plapt.score_candidates(proteins[0], molecules, batch_size=args.batch_size)
        elif len(proteins) == len(molecules):
            results = plapt.predict_affinity(proteins, molecules, batch_size=args.batch_size)
        else:
            print("Error: The number of proteins must be either 1 or equal to the number of molecules.")
            sys.exit(1)
    else:
        print("Error: At least one protein sequence must be provided.")
        sys.exit(1)
    
    write_output(results, args.output)

if __name__ == "__main__":
    main()