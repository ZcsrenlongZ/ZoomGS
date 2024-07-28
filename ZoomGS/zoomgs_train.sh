#!/bin/bash

echo "Start to train the model...."

dataroot="../dataset/ZoomGS_dataset/"
output="./ckpt/zoomgs/"

if [ ! -d "output" ]; then
        mkdir $output
fi

softfiles=$(ls $dataroot)
softfiles=("01") #  scene id

echo $softfiles
for sfile in $softfiles
do 
    scene=$dataroot$sfile
    outputdir=$output$sfile
    echo $scene
    echo $outputdir
    # #  pretrain base gs with uw image
    CUDA_VISIBLE_DEVICES=1 CUDA_LAUNCH_BLOCKING=1 python zoomgs_train.py \
    -s $scene -m $outputdir --iterations 30000 --eval --port 6014 --stage "uw_pretrain"
    # #  joint training base gs, camTrans module with uw and w image
    CUDA_VISIBLE_DEVICES=1 CUDA_LAUNCH_BLOCKING=1 python zoomgs_train.py \
    -s $scene -m $outputdir --iterations 30000 --eval --port 6014 --stage "uw2wide"
    # # test zoomGS results
    CUDA_VISIBLE_DEVICES=1 python zoomgs_test.py -m $outputdir -s $scene --iteration 30000 --target "cx"
    # # generate camera transition sequences
    CUDA_VISIBLE_DEVICES=1 python zoomgs_render.py -m $outputdir -s $scene --iteration 30000 --target "cx"
    
done
