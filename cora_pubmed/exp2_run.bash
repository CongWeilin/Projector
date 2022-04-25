# DATASET=Cora
# for SEED in 0 10 123 231 321; do
#     python exp2_graph_newton.py    --dataset $DATASET --seed $SEED --lam 1e-2 
#     python exp2_graph_newton.py    --dataset $DATASET --seed $SEED --lam 1e-2  --is_influence
#     python exp2_graph_projector.py --dataset $DATASET --seed $SEED
# done


# DATASET=Citeseer
# for SEED in 0 10 123 231 321; do
#     python exp2_graph_newton.py    --dataset $DATASET --seed $SEED --lam 1e-3 
#     python exp2_graph_newton.py    --dataset $DATASET --seed $SEED --lam 1e-3  --is_influence
#     python exp2_graph_projector.py --dataset $DATASET --seed $SEED
# done

# DATASET=Pubmed
# for SEED in 0 10 123 231 321; do
#     python exp2_graph_newton.py    --dataset $DATASET --seed $SEED --lam 1e-2 
#     python exp2_graph_newton.py    --dataset $DATASET --seed $SEED --lam 1e-2  --is_influence
#     python exp2_graph_projector.py --dataset $DATASET --seed $SEED
# done


