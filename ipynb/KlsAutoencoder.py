
from fastai import *
from fastai.basic_data import *
from fastai.basic_train import *
from fastai.tabular import *
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset

class KlsDataset(Dataset):
    """Kmer latent representation dataset"""
    def __init__(self, data,noise=0.):
        super().__init__()
        self.items = data.values if isinstance(data, pd.DataFrame) else data
        self.noise = noise

    def __len__(self):
        return len(self.items)
    def __getitem__(self, idx):
        item = self.items[idx,:]
        return (item if self.noise == 0. else self.mix_noise(item), item)

def wing(dims):
    layer_dims = list(zip(dims[:-1],dims[1:]))
    fcl = [nn.Linear(*x, bias=False) for x in layer_dims]
    relu = [nn.ReLU() for _ in range(len(fcl))]
    layers = np.asarray(list(zip(fcl, relu))).ravel()[:-1]
    return nn.Sequential(*layers)

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)


def print_weights(nlayer):
    print(list(net.encoder.modules())[nlayer].weight)

class KlsAutoEncoder (nn.Module):
    """Generic autoencoder"""
    def __init__(self, encoder_dims, decoder_dims):
        super().__init__(self)
        self.encoder = wing(encoder_dims)
        self.decoder = wing(decoder_dims)

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)

    def save_encoder(self,file:PathOrStr):
        torch.save(self.encoder.state_dict(), path)

class Encoder():
    """Encoder part of KlsAutoeEncoder ready for inference"""

    def __init__(self,file:PathOrStr,dims:Collection=[100,50,3]):
        e = wing(dims).double()
        e.load_state_dict(torch.load(file))
        e.eval()
        self.e = e

    def transform(self,data:Collection):
        """transform ```data``` to latent representaion"""
        return self.e.forward(tensor(data).double()).cpu().detach().numpy()
