
from models.npull import NPullNetwork

def get_protein_pipeline(**kwargs):
    sdf_model = NPullNetwork(**kwargs).to(kwargs["device"])

class Protein_pipeline(nn.Module):

    def __init__(self):
        super(Protein_pipeline).__init__()

    def forward():
        pass
