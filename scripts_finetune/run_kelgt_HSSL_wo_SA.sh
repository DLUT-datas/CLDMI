# python finetune.py --config base --model_path ./models/pretrained/base/base.pth  --dataset bace --data_path ./datasets/ --dataset_type classification --metric rocauc --split scaffold-0 --weight_decay 0 --dropout 0 --lr 3e-5

 
for seed_i in 1 #2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
do
python finetune_kelgt.py \
    --seed $seed_i \
    --config base \
    --model_path ./pretrained_KeLGT_HSSL_wo_SA/pretrained/base_dkj/base_20.pth \
    --save_path './result_finetuned_kelgt_HSSL_wo_SA_demo' \
    --dataset bbbp \
    --data_path ./datasets/ \
    --dataset_type classification \
    --metric roc-auc \
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
#     --model_path ./pretrained_KeLGT_HSSL_wo_SA/pretrained/base_dkj/base_20.pth \
#     --save_path './result_finetuned_kelgt_HSSL_wo_SA' \
#     --dataset bace \
#     --data_path ./datasets/ \
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
#     --model_path ./pretrained_KeLGT_HSSL_wo_SA/pretrained/base_dkj/base_20.pth \
#     --save_path './result_finetuned_kelgt_HSSL_wo_SA' \
#     --dataset clintox \
#     --data_path ./datasets/ \
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
#     --model_path ./pretrained_KeLGT_HSSL_wo_SA/pretrained/base_dkj/base_20.pth \
#     --save_path './result_finetuned_kelgt_HSSL_wo_SA' \
#     --dataset sider \
#     --data_path ./datasets/ \
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
#     --model_path ./pretrained_KeLGT_HSSL_wo_SA/pretrained/base_dkj/base_20.pth \
#     --save_path './result_finetuned_kelgt_HSSL_wo_SA' \
#     --dataset tox21 \
#     --data_path ./datasets/ \
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
#     --model_path ./pretrained_KeLGT_HSSL_wo_SA/pretrained/base_dkj/base_20.pth \
#     --save_path './result_finetuned_kelgt_HSSL_wo_SA' \
#     --dataset toxcast \
#     --data_path ./datasets/ \
#     --dataset_type classification \
#     --metric rocauc \
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
#     --model_path ./pretrained_KeLGT_HSSL_wo_SA/pretrained/base_dkj/base_20.pth \
#     --save_path './result_finetuned_kelgt_HSSL_wo_SA' \
#     --dataset esol \
#     --data_path ./datasets/ \
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
#     --model_path ./pretrained_KeLGT_HSSL_wo_SA/pretrained/base_dkj/base_20.pth \
#     --save_path './result_finetuned_kelgt_HSSL_wo_SA' \
#     --dataset freesolv \
#     --data_path ./datasets/ \
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
#     --model_path ./pretrained_KeLGT_HSSL_wo_SA/pretrained/base_dkj/base_20.pth \
#     --save_path './result_finetuned_kelgt_HSSL_wo_SA' \
#     --dataset lipo \
#     --data_path ./datasets/ \
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
#     --model_path ./pretrained_KeLGT_HSSL_wo_SA/pretrained/base_dkj/base_20.pth \
#     --save_path './result_finetuned_kelgt_HSSL_wo_SA' \
#     --dataset Ephrin \
#     --data_path ./datasets/ \
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
#     --model_path ./pretrained_KeLGT_HSSL_wo_SA/pretrained/base_dkj/base_20.pth \
#     --save_path './result_finetuned_kelgt_HSSL_wo_SA' \
#     --dataset COX-2 \
#     --data_path ./datasets/ \
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
#     --model_path ./pretrained_KeLGT_HSSL_wo_SA/pretrained/base_dkj/base_20.pth \
#     --save_path './result_finetuned_kelgt_HSSL_wo_SA' \
#     --dataset pdbbind_full \
#     --data_path ./datasets/ \
#     --dataset_type regression \
#     --metric rmse \
#     --split balanced_scaffold-$seed_i \
#     --weight_decay 0 \
#     --dropout 0 \
#     --lr 3e-5

# done