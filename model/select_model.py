from model.scDCC import scDCC
from model.Trans_scRNA import Trans_scRNA

from ast import arg
import torch

def select_model(args, input_size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = None
    if args.model == "scDCC":
        model = scDCC(input_dim=input_size, z_dim=32, n_clusters=8, 
                encodeLayer=[256, 64], decodeLayer=[64, 256], sigma=args.sigma, gamma=args.gamma,
                ml_weight=1., cl_weight=1., args= args).cuda()
    if args.model == "Trans_scRNA":
        model = Trans_scRNA(input_dim=input_size, z_dim= 32 , n_clusters= 8, 
                        sigma=args.sigma, gamma=args.gamma,args= args).cuda()
    
    return model