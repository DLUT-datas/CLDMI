
for seed_i in 1 #2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20   
do
python finetune_cmpn.py \
    --data_path ./datasets/bbbp/bbbp.csv \
    --dataset_type classification \
    --seed $seed_i \
    --split_type 'scaffold_balanced' \
    --save_dir ./result_TSM_finetuned_CMPN_T_w_Pt_S_wo_Pt_demo/bbbp \
    --checkpoint_path "./pretrained_TSM_CMPN_T_w_Pt_S_wo_Pt/molecule_graph_model_20.pth"\
    --load_from_TSM \
    --save_smiles_splits

done



# for seed_i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20   
# do
# python finetune_cmpn.py \
#     --data_path ./datasets/bace/bace.csv \
#     --dataset_type classification \
#     --seed $seed_i \
#     --split_type 'scaffold_balanced' \
#     --save_dir ./result_TSM_finetuned_CMPN_T_w_Pt_S_wo_Pt/bace \
#     --checkpoint_path "./pretrained_TSM_CMPN_T_w_Pt_S_wo_Pt/molecule_graph_model_20.pth"\
#     --load_from_TSM \
#     --save_smiles_splits

# done



# for seed_i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20   
# do
# python finetune_cmpn.py \
#     --data_path ./datasets/clintox/clintox.csv \
#     --dataset_type classification \
#     --seed $seed_i \
#     --split_type 'scaffold_balanced' \
#     --save_dir ./result_TSM_finetuned_CMPN_T_w_Pt_S_wo_Pt/clintox \
#     --checkpoint_path "./pretrained_TSM_CMPN_T_w_Pt_S_wo_Pt/molecule_graph_model_20.pth"\
#     --load_from_TSM \
#     --save_smiles_splits

# done


# for seed_i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20   
# do
# python finetune_cmpn.py \
#     --data_path ./datasets/sider/sider.csv \
#     --dataset_type classification \
#     --seed $seed_i \
#     --split_type 'scaffold_balanced' \
#     --save_dir ./result_TSM_finetuned_CMPN_T_w_Pt_S_wo_Pt/sider \
#     --checkpoint_path "./pretrained_TSM_CMPN_T_w_Pt_S_wo_Pt/molecule_graph_model_20.pth"\
#     --load_from_TSM \
#     --save_smiles_splits

# done


# for seed_i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20   
# do
# python finetune_cmpn.py \
#     --data_path ./datasets/tox21/tox21.csv \
#     --dataset_type classification \
#     --seed $seed_i \
#     --split_type 'scaffold_balanced' \
#     --save_dir ./result_TSM_finetuned_CMPN_T_w_Pt_S_wo_Pt/tox21 \
#     --checkpoint_path "./pretrained_TSM_CMPN_T_w_Pt_S_wo_Pt/molecule_graph_model_20.pth"\
#     --load_from_TSM \
#     --save_smiles_splits

# done



# for seed_i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20   
# do
# python finetune_cmpn.py \
#     --data_path ./datasets/toxcast/toxcast.csv \
#     --dataset_type classification \
#     --seed $seed_i \
#     --split_type 'scaffold_balanced' \
#     --save_dir ./result_TSM_finetuned_CMPN_T_w_Pt_S_wo_Pt/toxcast \
#     --checkpoint_path "./pretrained_TSM_CMPN_T_w_Pt_S_wo_Pt/molecule_graph_model_20.pth"\
#     --load_from_TSM \
#     --save_smiles_splits

# done




# for seed_i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20   
# do
# python finetune_cmpn.py \
#     --data_path ./datasets/esol/esol.csv \
#     --dataset_type regression \
#     --seed $seed_i \
#     --split_type 'scaffold_balanced' \
#     --save_dir ./result_TSM_finetuned_CMPN_T_w_Pt_S_wo_Pt/esol \
#     --checkpoint_path "./pretrained_TSM_CMPN_T_w_Pt_S_wo_Pt/molecule_graph_model_20.pth"\
#     --load_from_TSM \
#     --save_smiles_splits

# done



# for seed_i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20   
# do
# python finetune_cmpn.py \
#     --data_path ./datasets/freesolv/freesolv.csv \
#     --dataset_type regression \
#     --seed $seed_i \
#     --split_type 'scaffold_balanced' \
#     --save_dir ./result_TSM_finetuned_CMPN_T_w_Pt_S_wo_Pt/freesolv \
#     --checkpoint_path "./pretrained_TSM_CMPN_T_w_Pt_S_wo_Pt/molecule_graph_model_20.pth"\
#     --load_from_TSM \
#     --save_smiles_splits

# done



# for seed_i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20   
# do
# python finetune_cmpn.py \
#     --data_path ./datasets/lipo/lipo.csv \
#     --dataset_type regression \
#     --seed $seed_i \
#     --split_type 'scaffold_balanced' \
#     --save_dir ./result_TSM_finetuned_CMPN_T_w_Pt_S_wo_Pt/lipo \
#     --checkpoint_path "./pretrained_TSM_CMPN_T_w_Pt_S_wo_Pt/molecule_graph_model_20.pth"\
#     --load_from_TSM \
#     --save_smiles_splits

# done




# for seed_i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20   
# do
# python finetune_cmpn.py \
#     --data_path ./datasets/Ephrin/Ephrin.csv \
#     --dataset_type regression \
#     --seed $seed_i \
#     --split_type 'scaffold_balanced' \
#     --save_dir ./result_TSM_finetuned_CMPN_T_w_Pt_S_wo_Pt/Ephrin \
#     --checkpoint_path "./pretrained_TSM_CMPN_T_w_Pt_S_wo_Pt/molecule_graph_model_20.pth"\
#     --load_from_TSM \
#     --save_smiles_splits

# done



# for seed_i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20   
# do
# python finetune_cmpn.py \
#     --data_path ./datasets/COX-2/COX-2.csv \
#     --dataset_type regression \
#     --seed $seed_i \
#     --split_type 'scaffold_balanced' \
#     --save_dir ./result_TSM_finetuned_CMPN_T_w_Pt_S_wo_Pt/COX-2 \
#     --checkpoint_path "./pretrained_TSM_CMPN_T_w_Pt_S_wo_Pt/molecule_graph_model_20.pth"\
#     --load_from_TSM \
#     --save_smiles_splits

# done



# for seed_i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20   
# do
# python finetune_cmpn.py \
#     --data_path ./datasets/pdbbind_full/pdbbind_full.csv \
#     --dataset_type regression \
#     --seed $seed_i \
#     --split_type 'scaffold_balanced' \
#     --save_dir ./result_TSM_finetuned_CMPN_T_w_Pt_S_wo_Pt/pdbbind_full \
#     --checkpoint_path "./pretrained_TSM_CMPN_T_w_Pt_S_wo_Pt/molecule_graph_model_20.pth"\
#     --load_from_TSM \
#     --save_smiles_splits

# done