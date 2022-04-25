
# python exp1_graph_projector_v2.py --use_cross_entropy --recycle_steps 500 --epochs 8000 --hop_neighbors 25 --num_remove_nodes 0.1 
# python exp1_graph_projector_v2.py --use_cross_entropy --recycle_steps 500 --epochs 8000 --hop_neighbors 25 --num_remove_nodes 0.05
# python exp1_graph_projector_v2.py --use_cross_entropy --recycle_steps 500 --epochs 8000 --hop_neighbors 25 --num_remove_nodes 0.02

# python exp1_graph_projector_v2.py --use_cross_entropy --recycle_steps 500 --epochs 8000 --hop_neighbors 25 --num_remove_nodes 0.02

# python exp1_graph_projector_v2.py --use_cross_entropy --recycle_steps 500 --epochs 8000 --hop_neighbors 25 --num_remove_nodes 0.1 --use_adapt_gcs_x --use_adapt_gcs_y
# python exp1_graph_projector_v2.py --use_cross_entropy --recycle_steps 500 --epochs 8000 --hop_neighbors 25 --num_remove_nodes 0.05 --use_adapt_gcs_x --use_adapt_gcs_y
# python exp1_graph_projector_v2.py --use_cross_entropy --recycle_steps 500 --epochs 8000 --hop_neighbors 25 --num_remove_nodes 0.02 --use_adapt_gcs_x --use_adapt_gcs_y


# python exp1_graph_influence.py --regen_feats --regen_model --use_label --num_remove_nodes 0.02 --lam 1e-3
# python exp1_graph_influence.py --regen_feats --regen_model --use_label --num_remove_nodes 0.05 --lam 1e-3
# python exp1_graph_influence.py --regen_feats --regen_model --use_label --num_remove_nodes 0.10 --lam 1e-2

# python exp1_graph_fisher.py --regen_feats --regen_model --use_label --num_remove_nodes 0.02 --lam 1e-3
# python exp1_graph_fisher.py --regen_feats --regen_model --use_label --num_remove_nodes 0.05 --lam 1e-3
# python exp1_graph_fisher.py --regen_feats --regen_model --use_label --num_remove_nodes 0.10 --lam 1e-3


# SEED=10
# python exp3_graph_projector_v2.py --use_cross_entropy \
#                                       --recycle_steps 500 \
#                                       --epochs 8000 --hop_neighbors 25 \
#                                       --seed $SEED --continue_unlearn_step

# python exp3_GAT.py --seed 123 --continue_unlearn_step
python exp3_GAT.py --seed 10 --continue_unlearn_step
python exp3_GAT.py --seed 10 --continue_unlearn_step
python exp3_GAT.py --seed 10 --continue_unlearn_step
python exp3_GAT.py --seed 10 --continue_unlearn_step

python exp3_GAT.py --seed 231
python exp3_GAT.py --seed 231 --continue_unlearn_step
python exp3_GAT.py --seed 231 --continue_unlearn_step
python exp3_GAT.py --seed 231 --continue_unlearn_step
python exp3_GAT.py --seed 231 --continue_unlearn_step

# python exp3_SAGE.py --seed 0 
# python exp3_SAGE.py --seed 0 --continue_unlearn_step
# python exp3_SAGE.py --seed 0 --continue_unlearn_step
# python exp3_SAGE.py --seed 0 --continue_unlearn_step
# python exp3_SAGE.py --seed 0 --continue_unlearn_step

# python exp3_SAGE.py --seed 321 
# python exp3_SAGE.py --seed 321 --continue_unlearn_step
# python exp3_SAGE.py --seed 321 --continue_unlearn_step
# python exp3_SAGE.py --seed 321 --continue_unlearn_step
# python exp3_SAGE.py --seed 321 --continue_unlearn_step


# python exp3_graphsaint.py --seed 123 --continue_unlearn_step
# python exp3_graphsaint.py --seed 123 --continue_unlearn_step
# python exp3_graphsaint.py --seed 123 --continue_unlearn_step
    
for SEED in 231 0 321; do
#     python exp3_graph_projector_v2.py --use_cross_entropy \
#                                       --recycle_steps 500 \
#                                       --epochs 8000 --hop_neighbors 25 \
#                                       --seed $SEED
                                      
#     python exp3_graph_projector_v2.py --use_cross_entropy \
#                                       --recycle_steps 500 \
#                                       --epochs 8000 --hop_neighbors 25 \
#                                       --seed $SEED \
#                                       --use_adapt_gcs_x --use_adapt_gcs_y
    python exp3_graphsaint.py --seed $SEED
    python exp3_graphsaint.py --seed $SEED --continue_unlearn_step
    python exp3_graphsaint.py --seed $SEED --continue_unlearn_step
    python exp3_graphsaint.py --seed $SEED --continue_unlearn_step
done

