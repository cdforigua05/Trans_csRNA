CUDA_VISIBLE_DEVICES=5 python main.py --model scDCC --save_dir results/RetinaMacosko --name RetinaMacosko_scDCC \
--batch_size 256 --pretrain_epochs 300 --dataset RetinaMacosko --data_file ./datos/RetinaMacosko/Macosko_mouse_retina.h5 --n_clusters 39