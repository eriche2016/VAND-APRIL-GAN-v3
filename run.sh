# prepare data
if [ 1 -eq 0 ]; then
    # mvtec dataset
    python data/mvtec.py
    # visa dataset
    python data/visa.py
fi
############################################################
# MVTec
############################################################
# ### train on the MVTec AD dataset (and perform zero-shot test on visa) 
if [ 1 -eq 1 ]; then
    CUDA_VISIBLE_DEVICES=3 python train.py --dataset mvtec --train_data_path ./data/mvtec \
    --save_path ./exps/visa/vit_large_14_518 --config_path ./open_clip/model_configs/ViT-L-14-336.json --model ViT-L-14-336 \
    --features_list 6 12 18 24 --pretrained openai --image_size 518  --batch_size 8 --aug_rate 0.2 --print_freq 1 \
    --epoch 3 --save_freq 1
fi

# train on visa
if [ 1 -eq 0 ]; then
    python train.py --dataset visa --train_data_path ./data/visa \
    --save_path ./exps/mvtec/vit_large_14_518 --config_path ./open_clip/model_configs/ViT-L-14-336.json --model ViT-L-14-336 \
    --features_list 6 12 18 24 --pretrained openai --image_size 518  --batch_size 8 --print_freq 1 \
    --epoch 15 --save_freq 1
fi

###############################
# few shot experiments 
################################
if [ 1 -eq 1 ]; then 
    ### test on the VisA dataset
    python test.py --mode few_shot --dataset visa \
    --data_path ./data/visa --save_path ./results/visa/few_shot/4shot/seed42 \
    --config_path ./open_clip/model_configs/ViT-L-14-336.json --checkpoint_path ./exps/pretrained/mvtec_pretrained.pth \
    --model ViT-L-14-336 --features_list 6 12 18 24 --few_shot_features 6 12 18 24 \
    --pretrained openai --image_size 518 --k_shot 4 --seed 42
fi 

if [ 1 -eq 0 ]; then 
    ### test on the MVTec AD dataset
    python test.py --mode few_shot --dataset mvtec \
    --data_path ./data/mvtec --save_path ./results/mvtec/few_shot/4shot/seed42 \
    --config_path ./open_clip/model_configs/ViT-L-14-336.json --checkpoint_path ./exps/pretrained/visa_pretrained.pth \
    --model ViT-L-14-336 --features_list 6 12 18 24 --few_shot_features 6 12 18 24 \
    --pretrained openai --image_size 518 --k_shot 4 --seed 42
fi 


###################################
# zero shot test 
###################################
if [ 1 -eq 0 ]; then 
    ### test on the VisA dataset
    python test.py --mode zero_shot --dataset visa \
    --data_path ./data/visa --save_path ./results/visa/zero_shot \
    --config_path ./open_clip/model_configs/ViT-L-14-336.json --checkpoint_path ./exps/pretrained/mvtec_pretrained.pth \
    --model ViT-L-14-336 --features_list 6 12 18 24 --pretrained openai --image_size 518
fi 

if [ 1 -eq 0 ]; then  
    ### test on the MVTec AD dataset
    python test.py --mode zero_shot --dataset mvtec \
    --data_path ./data/mvtec --save_path ./results/mvtec/zero_shot \
    --config_path ./open_clip/model_configs/ViT-L-14-336.json --checkpoint_path ./exps/pretrained/visa_pretrained.pth \
    --model ViT-L-14-336 --features_list 6 12 18 24 --pretrained openai --image_size 518
fi 