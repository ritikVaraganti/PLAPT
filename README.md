# PLAPT: Protein-Ligand Affinity Prediction Using Pretrained Transformers

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/plapt-protein-ligand-binding-affinity/protein-ligand-affinity-prediction-on-csar)](https://paperswithcode.com/sota/protein-ligand-affinity-prediction-on-csar?p=plapt-protein-ligand-binding-affinity)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/plapt-protein-ligand-binding-affinity/protein-ligand-affinity-prediction-on-pdbbind)](https://paperswithcode.com/sota/protein-ligand-affinity-prediction-on-pdbbind?p=plapt-protein-ligand-binding-affinity)

PLAPT is a state-of-the-art tool for predicting protein-ligand binding affinity, crucial for accelerating drug discovery processes. Our model leverages transfer learning from pretrained transformers like ProtBERT and ChemBERTa to achieve high accuracy while requiring minimal computational resources.

## Key Features

- **Efficient Processing**: Extremely lightweight prediction module allows for incredibly high throughput affinity prediction with cached embeddings.
- **Transfer Learning**: Uses pretrained models to extract rich protein and molecule features.
- **Versatile Usage**: Uses just 1D protein and ligand sequences as strings for input. Has a command-line interface and Python API for easy integration into various workflows.
- **High Accuracy**: Achieves top performance on benchmark datasets.

[Read our preprint](https://doi.org/10.1101/2024.02.08.575577)

## Model Architecture

PLAPT uses a novel branching neural network architecture that efficiently integrates features from protein and ligand encoders to estimate binding affinities:

![PLAPT Architecture](https://github.com/trrt-good/WELP-PLAPT/blob/main/Diagrams/PLAPT.png)

This architecture allows PLAPT to process complex molecular information effectively and highly efficiently when coupled with caching.

## Quick Start

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/trrt-good/WELP-PLAPT.git
   cd WELP-PLAPT
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### Using PLAPT

PLAPT can be used via command line or integrated into Python scripts.

#### Command Line Interface

Predict affinity for a single protein and multiple ligands:

```bash
python plapt_cli.py -p "SEQUENCE" -m "SMILES1" "SMILES2" "SMILES3"
```

Predict affinities for multiple protein-ligand pairs:

```bash
python plapt_cli.py -p "SEQUENCE1" "SEQUENCE2" -m "SMILES1" "SMILES2"
```

Use files for input:

```bash
python plapt_cli.py -p proteins.txt -m molecules.txt
```

Save results to a file:

```bash
python plapt_cli.py -p "SEQUENCE" -m "SMILES1" "SMILES2" -o results.json
```

#### Python Integration

```python
from plapt import Plapt

plapt = Plapt()

# Predict affinity for a single protein and multiple ligands
protein = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
molecules = ["CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F", 
             "COC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F"]

results = plapt.score_candidates(protein, molecules)
print(results)

# Predict affinities for multiple protein-ligand pairs
proteins = ["SEQUENCE1", "SEQUENCE2"]
molecules = ["SMILES1", "SMILES2"]

results = plapt.predict_affinity(proteins, molecules)
print(results)
```

## Citation

If you use PLAPT in your research, please cite our paper:

```
@misc{rose2023plapt,
  title={PLAPT: Protein-Ligand Binding Affinity Prediction Using Pretrained Transformers},
  author={Tyler Rose, Nicolò Monti, Navvye Anand, Tianyu Shen},
  journal={bioRxiv},
  year={2023},
  url={https://www.biorxiv.org/content/10.1101/2024.02.08.575577v3},
  doi={10.1101/2024.02.08.575577}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.