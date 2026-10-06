# Removing the activation functions in the predictors of CMPN, KeLGT, and FGRF.  

for seed_i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 
do
python fusion_CLDMI.py \
    --data_path ./datasets \
    --dataset bbbp \
    --dataset_type classification \
    --seed $seed_i\
    --metric roc-auc \
    --split_type scaffold_balanced \
    --lgt_model_path ./pretrained_KeLGT_HSSL/pretrained/base_dkj/base_20.pth \
    --checkpoint_path "./pretrained_CMPN_PgKD_initialized_from_random/molecule_graph_model_best.pth"\
    --load_from_TSM \
    --save_path ./result_finetune_fused_w_3s_bs32_64_CLDMI_ps \
    --dropout 0 \
    --save_smiles_splits \
    --save_dir ./ckpt/CLDMI_ps \

done



# for seed_i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 
# do
# python fusion_CLDMI.py \
#     --data_path ./datasets \
#     --dataset bace \
#     --dataset_type classification \
#     --seed $seed_i\
#     --metric roc-auc \
#     --split_type scaffold_balanced \
#     --lgt_model_path ./pretrained_KeLGT_HSSL/pretrained/base_dkj/base_20.pth \
#     --checkpoint_path "./pretrained_CMPN_PgKD_initialized_from_random/molecule_graph_model_best.pth"\
#     --load_from_TSM \
#     --save_path ./result_finetune_fused_w_3s_bs32_64_CLDMI_ps \
#     --dropout 0\
#     --save_smiles_splits \
#     --save_dir ./ckpt/CLDMI_ps \

# done



# for seed_i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 
# # for seed_i in 6 
# do
# python fusion_CLDMI.py \
#     --data_path ./datasets \
#     --dataset clintox \
#     --dataset_type classification \
#     --seed $seed_i\
#     --metric roc-auc \
#     --split_type scaffold_balanced \
#     --lgt_model_path ./pretrained_KeLGT_HSSL/pretrained/base_dkj/base_20.pth \
#     --checkpoint_path "./pretrained_CMPN_PgKD_initialized_from_random/molecule_graph_model_best.pth"\
#     --load_from_TSM \
#     --save_path ./result_finetune_fused_w_3s_bs32_64_CLDMI_ps \
#     --dropout 0\
#     --save_smiles_splits \
#     --save_dir ./ckpt/CLDMI_ps \

# done



# for seed_i in 1 #2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 
# do
# python fusion_CLDMI.py \
#     --data_path ./datasets \
#     --dataset sider \
#     --dataset_type classification \
#     --seed $seed_i\
#     --metric roc-auc \
#     --split_type scaffold_balanced \
#     --lgt_model_path ./pretrained_KeLGT_HSSL/pretrained/base_dkj/base_20.pth \
#     --checkpoint_path "./pretrained_CMPN_PgKD_initialized_from_random/molecule_graph_model_best.pth"\
#     --load_from_TSM \
#     --save_path ./result_finetune_fused_w_3s_bs32_64_CLDMI_ps \
#     --dropout 0\
#     --save_smiles_splits \
#     --save_dir ./ckpt/CLDMI_ps \

# done




# for seed_i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 
# do
# python fusion_CLDMI.py \
#     --data_path ./datasets \
#     --dataset Tox21 \
#     --dataset_type classification \
#     --seed $seed_i\
#     --metric roc-auc \
#     --split_type scaffold_balanced \
#     --lgt_model_path ./pretrained_KeLGT_HSSL/pretrained/base_dkj/base_20.pth \
#     --checkpoint_path "./pretrained_CMPN_PgKD_initialized_from_random/molecule_graph_model_best.pth"\
#     --load_from_TSM \
#     --save_path ./result_finetune_fused_w_3s_bs32_64_CLDMI_ps \
#     --dropout 0\
#     --save_smiles_splits \
#     --save_dir ./ckpt/CLDMI_ps \

# done





# for seed_i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 
# do
# python fusion_CLDMI.py \
#     --data_path ./datasets \
#     --dataset ToxCast \
#     --dataset_type classification \
#     --seed $seed_i\
#     --metric roc-auc \
#     --split_type scaffold_balanced \
#     --lgt_model_path ./pretrained_KeLGT_HSSL/pretrained/base_dkj/base_20.pth \
#     --checkpoint_path "./pretrained_CMPN_PgKD_initialized_from_random/molecule_graph_model_best.pth"\
#     --load_from_TSM \
#     --save_path ./result_finetune_fused_w_3s_bs32_64_CLDMI_ps \
#     --dropout 0\
#     --save_smiles_splits \
#     --save_dir ./ckpt/CLDMI_ps \

# done


