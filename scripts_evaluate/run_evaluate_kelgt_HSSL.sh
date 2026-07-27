# The key parameter controlling the training or evaluation mode of the KeLGT branch is eval_kelgt.

# -------------------------------------------------------------------------- Evaluation mode --------------------------------------------------------------------------

 
# for seed_i in 1 #2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
# do
# python finetune_kelgt.py \
#     --seed $seed_i \
#     --config base \
#     --model_path ./pretrained_KeLGT_HSSL/pretrained/base_dkj/base_20.pth \
#     --save_path ./result_finetune_kelgt_HSSL_demo \
#     --dataset bbbp \
#     --data_path ./datasets/ \
#     --dataset_type classification \
#     --metric roc-auc \
#     --eval_kelgt \
#     --split balanced_scaffold-$seed_i \
#     --weight_decay 0 \
#     --dropout 0 \
#     --eval_kelgt \
#     --lr 3e-5

# done



# for seed_i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
# do
# python finetune_kelgt.py \
#     --seed $seed_i \
#     --config base \
#     --model_path ./pretrained_KeLGT_HSSL/pretrained/base_dkj/base_20.pth \
#     --save_path ./result_finetuned_kelgt_HSSL \
#     --dataset bace \
#     --data_path ./datasets/ \
#     --dataset_type classification \
#     --metric rocauc \
#     --eval_kelgt \
#     --split balanced_scaffold-$seed_i \
#     --weight_decay 0 \
#     --dropout 0 \
#     --lr 3e-5

# done



# for seed_i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
# do
# python finetune_kelgt.py \
#     --seed $seed_i \
#     --config base \
#     --model_path ./pretrained_KeLGT_HSSL/pretrained/base_dkj/base_20.pth \
#     --save_path ./result_finetuned_kelgt_HSSL \
#     --dataset clintox \
#     --data_path ./datasets/ \
#     --dataset_type classification \
#     --metric rocauc \
#     --eval_kelgt \
#     --split balanced_scaffold-$seed_i \
#     --weight_decay 0 \
#     --dropout 0 \
#     --lr 3e-5

# done



# for seed_i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
# do
# python finetune_kelgt.py \
#     --seed $seed_i \
#     --config base \
#     --model_path ./pretrained_KeLGT_HSSL/pretrained/base_dkj/base_20.pth \
#     --save_path ./result_finetuned_kelgt_HSSL \
#     --dataset sider \
#     --data_path ./datasets/ \
#     --dataset_type classification \
#     --metric rocauc \
#     --eval_kelgt \
#     --split balanced_scaffold-$seed_i \
#     --weight_decay 0 \
#     --dropout 0 \
#     --lr 3e-5

# done



# for seed_i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
# do
# python finetune_kelgt.py \
#     --seed $seed_i \
#     --config base \
#     --model_path ./pretrained_KeLGT_HSSL/pretrained/base_dkj/base_20.pth \
#     --save_path ./result_finetuned_kelgt_HSSL \
#     --dataset tox21 \
#     --data_path ./datasets/ \
#     --dataset_type classification \
#     --metric rocauc \
#     --eval_kelgt \
#     --split balanced_scaffold-$seed_i \
#     --weight_decay 0 \
#     --dropout 0 \
#     --lr 3e-5

# done



# for seed_i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
# do
# python finetune_kelgt.py \
#     --seed $seed_i \
#     --config base \
#     --model_path ./pretrained_KeLGT_HSSL/pretrained/base_dkj/base_20.pth \
#     --save_path ./result_finetuned_kelgt_HSSL \
#     --dataset toxcast \
#     --data_path ./datasets/ \
#     --dataset_type classification \
#     --metric rocauc \
#     --eval_kelgt \
#     --split scaffold-$seed_i \
#     --weight_decay 0 \
#     --dropout 0 \
#     --lr 3e-5

# done



# -----------------------------------------------------------
# for seed_i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
# do
# python finetune_kelgt.py \
#     --seed $seed_i \
#     --config base \
#     --model_path ./pretrained_KeLGT_HSSL/pretrained/base_dkj/base_20.pth \
#     --save_path ./result_finetuned_kelgt_HSSL \
#     --dataset esol \
#     --data_path ./datasets/ \
#     --dataset_type regression \
#     --metric rmse \
#     --eval_kelgt \
#     --split balanced_scaffold-$seed_i \
#     --weight_decay 0 \
#     --dropout 0 \
#     --lr 3e-5

# done



# for seed_i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
# do
# python finetune_kelgt.py \
#     --seed $seed_i \
#     --config base \
#     --model_path ./pretrained_KeLGT_HSSL/pretrained/base_dkj/base_20.pth \
#     --save_path ./result_finetuned_kelgt_HSSL \
#     --dataset freesolv \
#     --data_path ./datasets/ \
#     --dataset_type regression \
#     --metric rmse \
#     --eval_kelgt \
#     --split balanced_scaffold-$seed_i \
#     --weight_decay 0 \
#     --dropout 0 \
#     --lr 3e-5

# done



# for seed_i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
# do
# python finetune_kelgt.py \
#     --seed $seed_i \
#     --config base \
#     --model_path ./pretrained_KeLGT_HSSL/pretrained/base_dkj/base_20.pth \
#     --save_path ./result_finetuned_kelgt_HSSL \
#     --dataset lipo \
#     --data_path ./datasets/ \
#     --dataset_type regression \
#     --metric rmse \
#     --eval_kelgt \
#     --split balanced_scaffold-$seed_i \
#     --weight_decay 0 \
#     --dropout 0 \
#     --lr 3e-5

# done



# for seed_i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
# do
# python finetune_kelgt.py \
#     --seed $seed_i \
#     --config base \
#     --model_path ./pretrained_KeLGT_HSSL/pretrained/base_dkj/base_20.pth \
#     --save_path ./result_finetuned_kelgt_HSSL \
#     --dataset Ephrin \
#     --data_path ./datasets/ \
#     --dataset_type regression \
#     --metric rmse \
#     --eval_kelgt \
#     --split balanced_scaffold-$seed_i \
#     --weight_decay 0 \
#     --dropout 0 \
#     --lr 3e-5

# done



for seed_i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
do
python finetune_kelgt.py \
    --seed $seed_i \
    --config base \
    --model_path ./pretrained_KeLGT_HSSL/pretrained/base_dkj/base_20.pth \
    --save_path ./result_finetuned_kelgt_HSSL \
    --dataset COX-2 \
    --data_path ./datasets/ \
    --dataset_type regression \
    --metric rmse \
    --eval_kelgt \
    --split balanced_scaffold-$seed_i \
    --weight_decay 0 \
    --dropout 0 \
    --lr 3e-5

done



# for seed_i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
# do
# python finetune_kelgt.py \
#     --seed $seed_i \
#     --config base \
#     --model_path ./pretrained_KeLGT_HSSL/pretrained/base_dkj/base_20.pth \
#     --save_path ./result_finetuned_kelgt_HSSL \
#     --dataset pdbbind_full \
#     --data_path ./datasets/ \
#     --dataset_type regression \
#     --metric rmse \
#     --eval_kelgt \
#     --split balanced_scaffold-$seed_i \
#     --weight_decay 0 \
#     --dropout 0 \
#     --lr 3e-5

# done

