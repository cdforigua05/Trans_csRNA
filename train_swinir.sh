CUDA_VISIBLE_DEVICES=7 python main.py --model SwinIR --save_dir results/SwinIR_scRNA --name 10X_PBCM_BaselineSwinIR \
--batch_size 1 --pretrain_epochs 300  --ae_weights /media/SSD0/cdforigua/Trans_csRNA/results/10X_PBMC/10X_PBCM_BaselineSwinIR/pretrained/AE_weights_p0_1last.pth.tar