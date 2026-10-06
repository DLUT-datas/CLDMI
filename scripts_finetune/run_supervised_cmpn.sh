# The key parameter controlling the training or evaluation mode of the CMPN branch is eval_cmpn.

# `run_cmpn` and `run_cmpn_branch` control different return values for the CMPN.



# -------------------------------------------------------------------------- Training mode --------------------------------------------------------------------------
#
# for seed_i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20   
# do
# python finetune_cmpn.py \
#     --data_path ./datasets/bbbp/bbbp.csv \
#     --dataset_type classification \
#     --seed $seed_i \
#     --split_type 'scaffold_balanced' \
#     --save_dir ./result_supervised_CMPN2/bbbp \
#     --run_cmpn \
#     --save_smiles_splits

# done



# for seed_i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20   
# do
# python finetune_cmpn.py \
#     --data_path ./datasets/bace/bace.csv \
#     --dataset_type classification \
#     --seed $seed_i \
#     --split_type 'scaffold_balanced' \
#     --save_dir ./result_supervised_CMPN2/bace \
#     --run_cmpn \
#     --save_smiles_splits

# done



# for seed_i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20   
# do
# python finetune_cmpn.py \
#     --data_path ./datasets/clintox/clintox.csv \
#     --dataset_type classification \
#     --seed $seed_i \
#     --split_type 'scaffold_balanced' \
#     --save_dir ./result_supervised_CMPN2/clintox \
#     --run_cmpn \
#     --save_smiles_splits

# done


for seed_i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20   
do
python finetune_cmpn.py \
    --data_path ./datasets/sider/sider.csv \
    --dataset_type classification \
    --seed $seed_i \
    --split_type 'scaffold_balanced' \
    --save_dir ./result_supervised_CMPN2/sider \
    --run_cmpn \
    --save_smiles_splits

done


for seed_i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20   
do
python finetune_cmpn.py \
    --data_path ./datasets/tox21/tox21.csv \
    --dataset_type classification \
    --seed $seed_i \
    --split_type 'scaffold_balanced' \
    --save_dir ./result_supervised_CMPN2/tox21 \
    --run_cmpn \
    --save_smiles_splits

done



for seed_i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20   
do
python finetune_cmpn.py \
    --data_path ./datasets/toxcast/toxcast.csv \
    --dataset_type classification \
    --seed $seed_i \
    --split_type 'scaffold_balanced' \
    --save_dir ./result_supervised_CMPN2/toxcast \
    --run_cmpn \
    --save_smiles_splits

done




# for seed_i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20   
# do
# python finetune_cmpn.py \
#     --data_path ./datasets/esol/esol.csv \
#     --dataset_type regression \
#     --seed $seed_i \
#     --split_type 'scaffold_balanced' \
#     --save_dir ./result_supervised_CMPN2/esol \
#     --run_cmpn \
#     --save_smiles_splits

# done



# for seed_i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20   
# do
# python finetune_cmpn.py \
#     --data_path ./datasets/freesolv/freesolv.csv \
#     --dataset_type regression \
#     --seed $seed_i \
#     --split_type 'scaffold_balanced' \
#     --save_dir ./result_supervised_CMPN2/freesolv \
#     --run_cmpn \
#     --save_smiles_splits

# done



# for seed_i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20   
# do
# python finetune_cmpn.py \
#     --data_path ./datasets/lipo/lipo.csv \
#     --dataset_type regression \
#     --seed $seed_i \
#     --split_type 'scaffold_balanced' \
#     --save_dir ./result_supervised_CMPN2/lipo \
#     --run_cmpn \
#     --save_smiles_splits

# done




# for seed_i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20   
# do
# python finetune_cmpn.py \
#     --data_path ./datasets/Ephrin/Ephrin.csv \
#     --dataset_type regression \
#     --seed $seed_i \
#     --split_type 'scaffold_balanced' \
#     --save_dir ./result_supervised_CMPN2/Ephrin \
#     --run_cmpn \
#     --save_smiles_splits

# done



# for seed_i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20   
# do
# python finetune_cmpn.py \
#     --data_path ./datasets/COX-2/COX-2.csv \
#     --dataset_type regression \
#     --seed $seed_i \
#     --split_type 'scaffold_balanced' \
#     --save_dir ./result_supervised_CMPN2/COX-2 \
#     --run_cmpn \
#     --save_smiles_splits

# done



# for seed_i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20   
# do
# python finetune_cmpn.py \
#     --data_path ./datasets/pdbbind_full/pdbbind_full.csv \
#     --dataset_type regression \
#     --seed $seed_i \
#     --split_type 'scaffold_balanced' \
#     --save_dir ./result_supervised_CMPN2/pdbbind_full \
#     --checkpoint_path "./pretrained_CMPN_PgKD_initialized_from_random/molecule_graph_model_best.pth"\
#     --run_cmpn \
#     --save_smiles_splits

# done