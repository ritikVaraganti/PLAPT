import torch
from transformers import BertTokenizer, BertModel, RobertaTokenizer, RobertaModel
import re
import onnxruntime
import numpy as np
from typing import List, Dict, Union

class PredictionModule:
    def __init__(self, model_path: str = "models/affinity_predictor.onnx"):
        self.session = onnxruntime.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.mean = 6.51286529169358
        self.scale = 1.5614094578916633

    def convert_to_affinity(self, normalized: float) -> Dict[str, float]:
        neg_log10_affinity_M = float((normalized * self.scale) + self.mean)
        affinity_uM = float((10**6) * (10**(-neg_log10_affinity_M)))
        return {
            "neg_log10_affinity_M": neg_log10_affinity_M,
            "affinity_uM": affinity_uM
        }

    def predict(self, batch_data: np.ndarray) -> List[Dict[str, float]]:
        affinities = []
        for feature in batch_data:
            affinity_normalized = self.session.run(None, {self.input_name: [feature], 'TrainingMode': np.array(False)})[0][0][0]
            affinities.append(self.convert_to_affinity(affinity_normalized))
        return affinities

class Plapt:
    def __init__(self, prediction_module_path: str = "models/affinity_predictor.onnx", device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        self.prot_tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
        self.prot_encoder = BertModel.from_pretrained("Rostlab/prot_bert").to(self.device)
        
        self.mol_tokenizer = RobertaTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
        self.mol_encoder = RobertaModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1").to(self.device)
        
        self.prediction_module = PredictionModule(prediction_module_path)
        self.cache = {}

    @staticmethod
    def preprocess_sequence(seq: str) -> str:
        return " ".join(re.sub(r"[UZOB]", "X", seq))

    def tokenize_molecule(self, mol_smiles: Union[str, List[str]]) -> torch.Tensor:
        return self.mol_tokenizer(mol_smiles, padding=True, max_length=278, truncation=True, return_tensors='pt')

    def tokenize_protein(self, prot_seq: Union[str, List[str]]) -> torch.Tensor:
        preprocessed = [self.preprocess_sequence(seq) if isinstance(seq, str) else self.preprocess_sequence(seq[0]) for seq in prot_seq]
        return self.prot_tokenizer(preprocessed, padding=True, max_length=3200, truncation=True, return_tensors='pt')

    def encode_molecules(self, mol_smiles: List[str]) -> torch.Tensor:
        tokens = self.tokenize_molecule(mol_smiles)
        with torch.no_grad():
            return self.mol_encoder(**tokens.to(self.device)).pooler_output.cpu()

    def encode_proteins(self, prot_seqs: List[str]) -> torch.Tensor:
        tokens = self.tokenize_protein(prot_seqs)
        with torch.no_grad():
            return self.prot_encoder(**tokens.to(self.device)).pooler_output.cpu()

    @staticmethod
    def make_batches(iterable: List, n: int = 1):
        length = len(iterable)
        for ndx in range(0, length, n):
            yield iterable[ndx:min(ndx + n, length)]

    def predict_affinity(self, prot_seqs: List[str], mol_smiles: List[str], batch_size: int = 4) -> List[Dict[str, float]]:
        if len(prot_seqs) != len(mol_smiles):
            raise ValueError("The number of proteins and molecules must be the same.")

        affinities = []
        for prot_batch, mol_batch in zip(self.make_batches(prot_seqs, batch_size), self.make_batches(mol_smiles, batch_size)):
            prot_encodings = self.encode_proteins(prot_batch)
            mol_encodings = self.encode_molecules(mol_batch)

            features = torch.cat((prot_encodings, mol_encodings), dim=1).numpy()
            batch_affinities = self.prediction_module.predict(features)
            affinities.extend(batch_affinities)

        return affinities

    def score_candidates(self, target_protein: str, mol_smiles: List[str], batch_size: int = 4) -> List[Dict[str, float]]:
        target_encoding = self.encode_proteins([target_protein])
        affinities = []

        for mol_batch in self.make_batches(mol_smiles, batch_size):
            mol_encodings = self.encode_molecules(mol_batch)
            
            # Repeat the target protein encoding for each molecule in the batch
            repeated_target = target_encoding.repeat(len(mol_batch), 1)
            
            features = torch.cat((repeated_target, mol_encodings), dim=1).numpy()
            batch_affinities = self.prediction_module.predict(features)
            affinities.extend(batch_affinities)

        return affinities

# Example usage
if __name__ == "__main__":
    plapt = Plapt()
    
    # Example for predict_affinity
    proteins = ["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG", 
                "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"]
    molecules = ["CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F", 
                 "COC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F"]
    
    results = plapt.predict_affinity(proteins, molecules)
    print("Predict Affinity Results:", results)
    
    # Example for score_candidates
    target_protein = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
    candidate_molecules = ["CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F", 
                           "COC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F",
                           "CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F"]
    
    scores = plapt.score_candidates(target_protein, candidate_molecules)
    print("Score Candidates Results:", scores)