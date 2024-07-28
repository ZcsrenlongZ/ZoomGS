
# EDSC, IFRNet, RIFE, AMT, UPRNet, EMAVFI
CUDA_VISIBLE_DEVICES=0 python ./test_syn.py --model RIFE --log_dir ./ckpt/RIFE_finetuned \
                                            --dataset_dir ../dataset/DCSZ_dataset/DCSZ_syn 