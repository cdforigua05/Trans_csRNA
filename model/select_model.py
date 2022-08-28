from model.scDCC import scDCC

from ast import arg


def select_model(args, input_size):
    model = None
    if args.model == "scDCC":
        model = scDCC(input_dim=input_size, z_dim=32, n_clusters=8, 
                encodeLayer=[256, 64], decodeLayer=[64, 256], sigma=args.sigma, gamma=args.gamma,
                ml_weight=1., cl_weight=1., args= args).cuda()
    
    return model