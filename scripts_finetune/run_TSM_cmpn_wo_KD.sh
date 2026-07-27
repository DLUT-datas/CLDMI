# # 
for seed_i in 1 #2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20   
do
python finetune_cmpn.py \
    --data_path ./datasets/bbbp/bbbp.csv \
    --dataset_type classification \
    --seed $seed_i \
    --split_type 'scaffold_balanced' \
    --save_dir ./result_pretrained_CMPN_wo_KD_demo/bbbp \
    --checkpoint_path "./pretrained_CMPN_from_KEGGCL/onehot_MPN_0613_2249_19th_epoch.pkl"\
    --load_from_KEGGCL \
    --save_smiles_splits

done



# for seed_i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20   
# do
# python finetune_cmpn.py \
#     --data_path ./datasets/bace/bace.csv \
#     --dataset_type classification \
#     --seed $seed_i \
#     --split_type 'scaffold_balanced' \
#     --save_dir ./result_pretrained_CMPN_wo_KD/bace \
#     --checkpoint_path "./pretrained_CMPN_from_KEGGCL/onehot_MPN_0613_2249_19th_epoch.pkl"\
#     --load_from_KEGGCL \
#     --save_smiles_splits

# done



# for seed_i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20   
# for seed_i in 6 
# do
# python finetune_cmpn.py \
#     --data_path ./datasets/clintox/clintox.csv \
#     --dataset_type classification \
#     --seed $seed_i \
#     --split_type 'scaffold_balanced' \
#     --save_dir ./result_pretrained_CMPN_wo_KD/clintox \
#     --checkpoint_path "./pretrained_CMPN_from_KEGGCL/onehot_MPN_0613_2249_19th_epoch.pkl"\
#     --load_from_KEGGCL \
#     --save_smiles_splits

# done


# for seed_i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20   
# do
# python finetune_cmpn.py \
#     --data_path ./datasets/sider/sider.csv \
#     --dataset_type classification \
#     --seed $seed_i \
#     --split_type 'scaffold_balanced' \
#     --save_dir ./result_pretrained_CMPN_wo_KD/sider \
#     --checkpoint_path "./pretrained_CMPN_from_KEGGCL/onehot_MPN_0613_2249_19th_epoch.pkl"\
#     --load_from_KEGGCL \
#     --save_smiles_splits

# done


# for seed_i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20   
# do
# python finetune_cmpn.py \
#     --data_path ./datasets/tox21/tox21.csv \
#     --dataset_type classification \
#     --seed $seed_i \
#     --split_type 'scaffold_balanced' \
#     --save_dir ./result_pretrained_CMPN_wo_KD/tox21 \
#     --checkpoint_path "./pretrained_CMPN_from_KEGGCL/onehot_MPN_0613_2249_19th_epoch.pkl"\
#     --load_from_KEGGCL \
#     --save_smiles_splits

# done



# for seed_i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20   
# do
# python finetune_cmpn.py \
#     --data_path ./datasets/toxcast/toxcast.csv \
#     --dataset_type classification \
#     --seed $seed_i \
#     --split_type 'scaffold_balanced' \
#     --save_dir ./result_pretrained_CMPN_wo_KD/toxcast \
#     --checkpoint_path "./pretrained_CMPN_from_KEGGCL/onehot_MPN_0613_2249_19th_epoch.pkl"\
#     --load_from_KEGGCL \
#     --save_smiles_splits

# done




# for seed_i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20   
# do
# python finetune_cmpn.py \
#     --data_path ./datasets/esol/esol.csv \
#     --dataset_type regression \
#     --seed $seed_i \
#     --split_type 'scaffold_balanced' \
#     --save_dir ./result_pretrained_CMPN_wo_KD/esol \
#     --checkpoint_path "./pretrained_CMPN_from_KEGGCL/onehot_MPN_0613_2249_19th_epoch.pkl"\
#     --load_from_KEGGCL \
#     --save_smiles_splits

# done



# for seed_i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20   
# do
# python finetune_cmpn.py \
#     --data_path ./datasets/freesolv/freesolv.csv \
#     --dataset_type regression \
#     --seed $seed_i \
#     --split_type 'scaffold_balanced' \
#     --save_dir ./result_pretrained_CMPN_wo_KD/freesolv \
#     --checkpoint_path "./pretrained_CMPN_from_KEGGCL/onehot_MPN_0613_2249_19th_epoch.pkl"\
#     --load_from_KEGGCL \
#     --save_smiles_splits

# done


# 
# for seed_i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20   
# do
# python finetune_cmpn.py \
#     --data_path ./datasets/lipo/lipo.csv \
#     --dataset_type regression \
#     --seed $seed_i \
#     --split_type 'scaffold_balanced' \
#     --save_dir ./result_pretrained_CMPN_wo_KD/lipo \
#     --checkpoint_path "./pretrained_CMPN_from_KEGGCL/onehot_MPN_0613_2249_19th_epoch.pkl"\
#     --load_from_KEGGCL \
#     --save_smiles_splits

# done




# for seed_i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20   
# do
# python finetune_cmpn.py \
#     --data_path ./datasets/Ephrin/Ephrin.csv \
#     --dataset_type regression \
#     --seed $seed_i \
#     --split_type 'scaffold_balanced' \
#     --save_dir ./result_pretrained_CMPN_wo_KD/Ephrin \
#     --checkpoint_path "./pretrained_CMPN_from_KEGGCL/onehot_MPN_0613_2249_19th_epoch.pkl"\
#     --load_from_KEGGCL \
#     --save_smiles_splits

# done



# for seed_i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20   
# do
# python finetune_cmpn.py \
#     --data_path ./datasets/COX-2/COX-2.csv \
#     --dataset_type regression \
#     --seed $seed_i \
#     --split_type 'scaffold_balanced' \
#     --save_dir ./result_pretrained_CMPN_wo_KD/COX-2 \
#     --checkpoint_path "./pretrained_CMPN_from_KEGGCL/onehot_MPN_0613_2249_19th_epoch.pkl"\
#     --load_from_KEGGCL \
#     --save_smiles_splits

# done



# for seed_i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20   
# do
# python finetune_cmpn.py \
#     --data_path ./datasets/pdbbind_full/pdbbind_full.csv \
#     --dataset_type regression \
#     --seed $seed_i \
#     --split_type 'scaffold_balanced' \
#     --save_dir ./result_pretrained_CMPN_wo_KD/pdbbind_full \
#     --checkpoint_path "./pretrained_CMPN_from_KEGGCL/onehot_MPN_0613_2249_19th_epoch.pkl"\
#     --load_from_KEGGCL \
#     --save_smiles_splits

# done