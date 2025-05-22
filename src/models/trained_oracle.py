import pickle
from typing import List
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from mhfp.encoder import MHFPEncoder
import joblib

class TrainedOracle:
    def __init__(self):
        model_path=r"C:\Users\crist\Downloads\P35462_model.pkl"
        with open(model_path, 'rb') as f:
            self.model = joblib.load(f)
        self.encoder = MHFPEncoder()

    def _smiles_to_secfp(self, smiles_list, n_bits = 4096, radius = 2):
        fingerprints = []
        for smi in smiles_list:
            encoding = self.encoder.secfp_from_smiles(smi, length=n_bits, radius=radius)
            fingerprints.append(encoding)
        return fingerprints

    def score(self, smiles_list: List[str]) -> np.ndarray:
        fps = self._smiles_to_secfp(smiles_list)
        X = np.array(fps)
        return self.model.predict(X)