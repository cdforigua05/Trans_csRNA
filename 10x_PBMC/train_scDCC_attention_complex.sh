CUDA_VISIBLE_DEVICES=4 python main.py --model scDCC --save_dir results/10X_PBMC --name 10X_PBCM_scDCC_attention_complex \
--batch_size 256 --pretrain_epochs 300 --complex_attention