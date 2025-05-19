import torch
import sys
import pathlib
from pathlib import Path

FILE_PATH = str(Path(__file__).parent.resolve())
FILE_PATH = "/".join(FILE_PATH.split("\\"))
print(FILE_PATH)
REPO_PATH = "."
sys.path.append(REPO_PATH)
Path(REPO_PATH+'/outputs').mkdir(parents=True, exist_ok=True)

from src.models.ddpm import LatentDiffusion
from src.models.components.vae import PropertyPredictor

################################################################################################
# set model parameters

input_dim = 1024 # do not change
mlp_dims = 2048 # do not change
num_sample_mols = 50 # how many molecules to generate
uncond_dm_ckpt = REPO_PATH + '/model/drugdiff.ckpt'
output_file = REPO_PATH+'/outputs/zero_guidance_generation_50'

################################################################################################
# load DrugDiff

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
model = LatentDiffusion(input_dim = input_dim,
                        output_dim = input_dim,
                        mlp_dims = mlp_dims,
                        num_sample_mols = num_sample_mols,
                        vae_ckpt = None,
                        uncond_dm_ckpt = uncond_dm_ckpt)

model.to(device)
model.eval()

################################################################################################
# load classifier (here as an example: molecular weight)

classifier_mw = PropertyPredictor.load_from_checkpoint(REPO_PATH + '/model/molecular_weight.ckpt', in_dim=model.max_len*len(model.symbol_to_idx))
classifier_mw.eval()
classifier_mw = classifier_mw.to(device)

################################################################################################
# generate

model.apply_guidance(classifiers = [classifier_mw], 
                    # negative weights decrease the property value, positive ones increase it
                    # in case of multi-property generation, the ratio between the property weights
                    # influences how strongly they are considered, e.g.
                    # propA has weight 1 and propB has weight 2 than propB is guided for twice as strongly.
                    weights = [-1], 
                    properties = ['bioactivity'],
                    classifier_scales = [0, 150], # we include 0 (unguided) to have a reference for evaluation
                    sample_num = num_sample_mols, # how many molecules to generate (as defined above)
                    exp_name = output_file)