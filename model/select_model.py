from model.scDCC import scDCC
from model.Trans_scRNA import Trans_scRNA
from model.SwinIR_scRNA import SwinIR
from model.scDCCRes import scDCCRes
from ast import arg
import torch

def select_model(args, input_size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = None
    if args.model == "scDCC":
        model = scDCC(input_dim=input_size, z_dim=32, n_clusters=args.n_clusters, 
                encodeLayer=[256,64], decodeLayer=[64,256], sigma=args.sigma, gamma=args.gamma,
                ml_weight=1., cl_weight=1., args= args).cuda()
    if args.model == "scDCCRes":
        model = scDCCRes(input_dim=input_size, z_dim=32, n_clusters=8, 
                encodeLayer=[256, 64], decodeLayer=[64, 256], sigma=args.sigma, gamma=args.gamma,
                ml_weight=1., cl_weight=1., args= args).cuda()
    if args.model == "SwinIR":
        model = SwinIR(upscale=1, in_chans=1, img_size=128, window_size=8,
                    img_range=1., depths=[6], embed_dim=180, num_heads=[6],
                    mlp_ratio=2, upsampler='', resi_connection='1conv', input_dim=input_size, z_dim=32, sigma=args.sigma
                    , args=args)
    return model