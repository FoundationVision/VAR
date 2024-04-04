import os

import torch

from models.vae.vqvae import DiscreteVAE
from models.vae.quant import VectorQuantizer2


if __name__ == '__main__':
    seed = 0
    torch.backends.cudnn.deterministic = True
    import random, numpy
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    xx = torch.rand(2, 3, 32, 32)
    v = DiscreteVAE(vocab_size=16384, z_channels=128, ch=128)
    v: DiscreteVAE
    DiscreteVAE.forward
    rec1, idx1, loss1, rec2, idx2, loss2 = v(xx)
    (rec1.mean()-rec2.mean() + loss1+loss2).backward()
    print(rec1.data.view(-1)[:5])
    print(v.encoder.conv_in.weight.grad.view(-1)[:5])
"""
tensor([0.7631, 1.2052, 1.2993, 1.2156, 0.6183])
tensor([ 0.0015,  0.0026,  0.0028, -0.0003,  0.0021])
"""