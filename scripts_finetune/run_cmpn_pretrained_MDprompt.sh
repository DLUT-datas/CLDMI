# The key parameter controlling the training or evaluation mode of the CMPN branch is eval_cmpn.

# `run_cmpn` and `run_cmpn_branch` control different return values for the CMPN.



# # -------------------------------------------------------------------------- Training mode --------------------------------------------------------------------------
# for seed_i in 1 #2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20   
# do
# python finetune_cmpn.py \
#     --data_path ./datasets/bbbp/bbbp.csv \
#     --dataset_type classification \
#     --seed $seed_i \
#     --split_type 'scaffold_balanced' \
#     --save_dir ./result_finetuned_CMPN_only_MDprompt/bbbp \
#     --checkpoint_path "./pretrained_CMPN_from_only_MDprompt/molecule_graph_model_best.pth"\
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
#     --save_dir ./result_finetuned_CMPN_only_MDprompt/bace \
#     --checkpoint_path "./pretrained_CMPN_from_only_MDprompt/molecule_graph_model_best.pth"\
#     --run_cmpn \
#     --save_smiles_splits

# done



# for seed_i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20   
# # for seed_i in 6 
# do
# python finetune_cmpn.py \
#     --data_path ./datasets/clintox/clintox.csv \
#     --dataset_type classification \
#     --seed $seed_i \
#     --split_type 'scaffold_balanced' \
#     --save_dir ./result_finetuned_CMPN_only_MDprompt/clintox \
#     --checkpoint_path "./pretrained_CMPN_from_only_MDprompt/molecule_graph_model_best.pth"\
#     --run_cmpn \
#     --save_smiles_splits

# done


# for seed_i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20   
# do
# python finetune_cmpn.py \
#     --data_path ./datasets/sider/sider.csv \
#     --dataset_type classification \
#     --seed $seed_i \
#     --split_type 'scaffold_balanced' \
#     --save_dir ./result_finetuned_CMPN_only_MDprompt/sider \
#     --checkpoint_path "./pretrained_CMPN_from_only_MDprompt/molecule_graph_model_best.pth"\
#     --run_cmpn \
#     --save_smiles_splits

# done


for seed_i in 1 #2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20   
do
python finetune_cmpn.py \
    --data_path ./datasets/tox21/tox21.csv \
    --dataset_type classification \
    --seed $seed_i \
    --split_type 'scaffold_balanced' \
    --save_dir ./result_finetuned_CMPN_only_MDprompt/tox21 \
    --checkpoint_path "./pretrained_CMPN_from_only_MDprompt/molecule_graph_model_best.pth"\
    --run_cmpn \
    --save_smiles_splits

done



for seed_i in 1 #2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20   
do
python finetune_cmpn.py \
    --data_path ./datasets/toxcast/toxcast.csv \
    --dataset_type classification \
    --seed $seed_i \
    --split_type 'scaffold_balanced' \
    --save_dir ./result_finetuned_CMPN_only_MDprompt/toxcast \
    --checkpoint_path "./pretrained_CMPN_from_only_MDprompt/molecule_graph_model_best.pth"\
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
#     --save_dir ./result_finetuned_CMPN_only_MDprompt/esol \
#     --checkpoint_path "./pretrained_CMPN_from_only_MDprompt/molecule_graph_model_best.pth"\
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
#     --save_dir ./result_finetuned_CMPN_only_MDprompt/freesolv \
#     --checkpoint_path "./pretrained_CMPN_from_only_MDprompt/molecule_graph_model_best.pth"\
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
#     --save_dir ./result_finetuned_CMPN_only_MDprompt/lipo \
#     --checkpoint_path "./pretrained_CMPN_from_only_MDprompt/molecule_graph_model_best.pth"\
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
#     --save_dir ./result_finetuned_CMPN_only_MDprompt/Ephrin \
#     --checkpoint_path "./pretrained_CMPN_from_only_MDprompt/molecule_graph_model_best.pth"\
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
#     --save_dir ./result_finetuned_CMPN_only_MDprompt/COX-2 \
#     --checkpoint_path "./pretrained_CMPN_from_only_MDprompt/molecule_graph_model_best.pth"\
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
#     --save_dir ./result_finetuned_CMPN_only_MDprompt/pdbbind_full \
#     --checkpoint_path "./pretrained_CMPN_from_only_MDprompt/molecule_graph_model_best.pth"\
#     --run_cmpn \
#     --save_smiles_splits

# done