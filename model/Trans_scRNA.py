from turtle import forward
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model.layers import ZINBLoss, MeanAct, DispAct
import numpy as np
from sklearn.cluster import KMeans
import math, os
from sklearn import metrics
from utils import cluster_acc
import tqdm

class Trans_scRNA(nn.Module):
    def __init__(self, input_dim, z_dim, n_clusters, num_layer=6, nhead=8, args=None,
                sigma = 1., gamma = 1., alpha = 1.) -> None:
        super(Trans_scRNA, self).__init__()
        self.z_dim = z_dim
        self.n_cluster = n_clusters
        self.input_dim = input_dim
        self.num_layer = num_layer
        self.nhead = nhead
        self.sigma = sigma 
        self.gamma = gamma
        self.alpha = alpha
        self.embed = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim*3),
            nn.ReLU()
        )
        self.conv1 = nn.Conv1d(1, self.z_dim, 3, stride=3)
        self.positional_encoding = torch.tensor(self.positional_encoding(self.input_dim, self.z_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.z_dim, nhead=self.nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layer)

        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=self.z_dim, nhead=self.nhead)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.num_layer)

        self._dec_mean = nn.Sequential(nn.Conv1d(32, 1, 1), MeanAct())
        self._dec_disp = nn.Sequential(nn.Conv1d(32, 1, 1), DispAct())
        self._dec_pi = nn.Sequential(nn.Conv1d(32, 1, 1), nn.Sigmoid())

        self.mu = Parameter(torch.Tensor(n_clusters, z_dim))
        self.zinb_loss = ZINBLoss().cuda()
        self.args = args

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        self.load_state_dict(model_dict)
    
    def soft_assign(self, z):
        q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - self.mu)**2, dim=2) / self.alpha)
        q = q**((self.alpha+1.0)/2.0)
        q = (q.t() / torch.sum(q, dim=1)).t()
        return q
    
    def target_distribution(self, q):
        p = q**2 / q.sum(0)
        return (p.t() / p.sum(1)).t()

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        return pos * angle_rates

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis],
                                np.arange(d_model)[np.newaxis, :],
                                d_model)
        
        # apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        
        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
            
        pos_encoding = angle_rads[np.newaxis, ...]
            
        return pos_encoding
    
    def forward(self, x):
        h = self.conv1(self.embed(x+torch.rand_like(x)*self.sigma).unsqueeze(1)).transpose(2,1)
        h = h + self.positional_encoding.to(h.device)
        z =  self.transformer_encoder(h.float())
        d = self.transformer_decoder(z.float(),z.float()).transpose(2,1)
        __mean = self._dec_mean(d).squeeze(1)
        __disp = self._dec_disp(d).squeeze(1)
        __pi = self._dec_pi(d).squeeze(1)
        # Representaciones normales sin el ruido
        h0 = self.conv1(self.embed(x).unsqueeze(1)).transpose(2,1) + self.positional_encoding.to(x.device)
        z0 = self.transformer_encoder(h0.float())
        q = self.soft_assign(z0.mean(1)) #TODO: revisar esto
        return z0, q, __mean, __disp, __pi
    
    def encodeBatch(self, X, batch_size=256):
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()
        
        encoded = []
        num = X.shape[0]
        num_batch = int(math.ceil(1.0*X.shape[0]/batch_size))
        for batch_idx in range(num_batch):
            xbatch = X[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
            inputs = Variable(xbatch)
            z,_, _, _, _ = self.forward(inputs)
            encoded.append(z.data)

        encoded = torch.cat(encoded, dim=0)
        return encoded
    
    def cluster_loss(self, p, q):
        def kld(target, pred):
            return torch.mean(torch.sum(target*torch.log(target/(pred+1e-6)), dim=-1))
        kldloss = kld(p, q)
        return self.gamma*kldloss
    
    def pretrain_autoencoder(self, x, X_raw, size_factor, batch_size=256, lr=0.001, epochs=400, ae_save=True, ae_weights='AE_weights.pth.tar'):
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()
        dataset = TensorDataset(torch.Tensor(x), torch.Tensor(X_raw), torch.Tensor(size_factor))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        print("Pretraining stage")
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, amsgrad=True)
        loss_best = float('inf')
        for epoch in range(epochs):
            loss_avg = 0
            for batch_idx, (x_batch, x_raw_batch, sf_batch) in enumerate(dataloader):
                x_tensor = Variable(x_batch).cuda()
                x_raw_tensor = Variable(x_raw_batch).cuda()
                sf_tensor = Variable(sf_batch).cuda()
                _, _, mean_tensor, disp_tensor, pi_tensor = self.forward(x_tensor)
                loss = self.zinb_loss(x=x_raw_tensor, mean=mean_tensor, disp=disp_tensor, pi=pi_tensor, scale_factor=sf_tensor)
                loss_avg += loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print('Pretrain epoch [{}/{}], ZINB loss:{:.4f}'.format(batch_idx+1, epoch+1, loss.item()))
            loss_avg = loss_avg/(batch_idx+1)
            if loss_avg<loss_best:
                loss_best= loss_avg
                torch.save({'ae_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                    os.path.join("results", self.args.dataset, self.args.name, "pretrained", ae_weights))
            print('Epoch {}, ZINB loss Epoch: {:.4f}, ZINB loss Best: {:.4f}'.format(epoch+1, loss_avg, loss_best))
        if ae_save:
            torch.save({'ae_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                    os.path.join("results", self.args.dataset, self.args.name, "pretrained", ae_weights.replace(".pth.tar","last.pth.tar")))

    def save_checkpoint(self, state, index, filename):
        newfilename = os.path.join(filename, 'FTcheckpoint_%d.pth.tar' % index)
        torch.save(state, newfilename)



