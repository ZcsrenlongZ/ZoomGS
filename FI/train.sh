# EDSC, IFRNet, RIFE, AMT, UPRNet, EMAVFI
CUDA_VISIBLE_DEVICES=1,4 python -m torch.distributed.launch --nproc_per_node=2 --master_port=29502 train.py \
                                --model RIFE\
                                --log_dir ./ckpt/RIFE_finetuned_recode \
                                --dataset_dir ../dataset/DCSZ_dataset/DCSZ_syn \
                                --epoch 100 \
                                --world_size=2