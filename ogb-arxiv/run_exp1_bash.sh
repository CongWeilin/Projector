
# python exp1_graph_influence.py --num_remove_nodes 0.02
# python exp1_graph_influence.py --num_remove_nodes 0.05
# python exp1_graph_influence.py --num_remove_nodes 0.10

# python exp1_graph_fisher.py --num_remove_nodes 0.02
# python exp1_graph_fisher.py --num_remove_nodes 0.05
# python exp1_graph_fisher.py --num_remove_nodes 0.10

# python exp1_graph_eraser.py --parallel_unlearning 1 --hop_neighbors 15
# python exp1_graph_projector.py --parallel_unlearning 1

# for SEED in 0 10 123 231 321; do
#     python exp3_graph_projector.py --parallel_unlearning 20 \
#                                    --num_remove_nodes 0.2 \
#                                    --seed $SEED \
#                                    --use_cross_entropy \
#                                    --use_adapt_gcs
# done

# for SEED in 0 10 123 231 321; do
#     python exp3_graph_projector.py --parallel_unlearning 20 \
#                                    --num_remove_nodes 0.2 \
#                                    --seed $SEED \
#                                    --use_cross_entropy 
# done


