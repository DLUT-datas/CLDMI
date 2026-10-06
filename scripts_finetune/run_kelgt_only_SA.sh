# python finetune.py --config base --model_path ../models/pretrained/base/base.pth  --dataset bace --data_path ../datasets/ --dataset_type classification --metric rocauc --split scaffold-0 --weight_decay 0 --dropout 0 --lr 3e-5

# 
# for seed_i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
# do
# python finetune_kelgt.py \
#     --seed $seed_i \
#     --config base \
#     --model_path ../models_only_SA/pretrained/base_dkj/base_15.pth \
#     --save_path '../results_finetune_kelgt_Pret_ablation_only_SA_b256_e20' \
#     --dataset bbbp \
#     --data_path ../datasets/ \
#     --dataset_type classification \
#     --metric rocauc \
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
#     --model_path ../models_only_SA/pretrained/base_dkj/base_15.pth \
#     --save_path '../results_finetune_kelgt_Pret_ablation_only_SA_b256_e20' \
#     --dataset bace \
#     --data_path ../datasets/ \
#     --dataset_type classification \
#     --metric rocauc \
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
#     --model_path ../models_only_SA/pretrained/base_dkj/base_15.pth \
#     --save_path '../results_finetune_kelgt_Pret_ablation_only_SA_b256_e20' \
#     --dataset clintox \
#     --data_path ../datasets/ \
#     --dataset_type classification \
#     --metric rocauc \
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
#     --model_path ../models_only_SA/pretrained/base_dkj/base_15.pth \
#     --save_path '../results_finetune_kelgt_Pret_ablation_only_SA_b256_e20' \
#     --dataset sider \
#     --data_path ../datasets/ \
#     --dataset_type classification \
#     --metric rocauc \
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
#     --model_path ../models_only_SA/pretrained/base_dkj/base_15.pth \
#     --save_path '../results_finetune_kelgt_Pret_ablation_only_SA_b256_e20' \
#     --dataset tox21 \
#     --data_path ../datasets/ \
#     --dataset_type classification \
#     --metric rocauc \
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
#     --model_path ../models_only_SA/pretrained/base_dkj/base_15.pth \
#     --save_path '../results_finetune_kelgt_Pret_ablation_only_SA_b256_e20' \
#     --dataset toxcast \
#     --data_path ../datasets/ \
#     --dataset_type classification \
#     --metric rocauc \
#     --split scaffold-$seed_i \
#     --weight_decay 0 \
#     --dropout 0 \
#     --lr 3e-5

# done



# # -----------------------------------------------------------
# for seed_i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
# do
# python finetune_kelgt.py \
#     --seed $seed_i \
#     --config base \
#     --model_path ../models_only_SA/pretrained/base_dkj/base_15.pth \
#     --save_path '../results_finetune_kelgt_Pret_ablation_only_SA_b256_e20_demo' \
#     --dataset esol \
#     --data_path ../datasets/ \
#     --dataset_type regression \
#     --metric rmse \
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
#     --model_path ../models_only_SA/pretrained/base_dkj/base_15.pth \
#     --save_path '../results_finetune_kelgt_Pret_ablation_only_SA_b256_e20' \
#     --dataset freesolv \
#     --data_path ../datasets/ \
#     --dataset_type regression \
#     --metric rmse \
#     --split balanced_scaffold-$seed_i \
#     --weight_decay 0 \
#     --dropout 0 \
#     --lr 3e-5

# done


#  
# for seed_i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
# do
# python finetune_kelgt.py \
#     --seed $seed_i \
#     --config base \
#     --model_path ../models_only_SA/pretrained/base_dkj/base_15.pth \
#     --save_path '../results_finetune_kelgt_Pret_ablation_only_SA_b256_e20' \
#     --dataset lipo \
#     --data_path ../datasets/ \
#     --dataset_type regression \
#     --metric rmse \
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
#     --model_path ../models_only_SA/pretrained/base_dkj/base_15.pth \
#     --save_path '../results_finetune_kelgt_Pret_ablation_only_SA_b256_e20' \
#     --dataset Ephrin \
#     --data_path ../datasets/ \
#     --dataset_type regression \
#     --metric rmse \
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
    --model_path ../models_only_SA/pretrained/base_dkj/base_15.pth \
    --save_path '../results_finetune_kelgt_Pret_ablation_only_SA_b256_e20' \
    --dataset COX-2 \
    --data_path ../datasets/ \
    --dataset_type regression \
    --metric rmse \
    --split balanced_scaffold-$seed_i \
    --weight_decay 0 \
    --dropout 0 \
    --lr 3e-5

done



for seed_i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
do
python finetune_kelgt.py \
    --seed $seed_i \
    --config base \
    --model_path ../models_only_SA/pretrained/base_dkj/base_15.pth \
    --save_path '../results_finetune_kelgt_Pret_ablation_only_SA_b256_e20' \
    --dataset pdbbind_full \
    --data_path ../datasets/ \
    --dataset_type regression \
    --metric rmse \
    --split balanced_scaffold-$seed_i \
    --weight_decay 0 \
    --dropout 0 \
    --lr 3e-5

done