python -m test.other.link_prediction --embedding_type="poincare_maps" --q=0.9 --dataset="ToggleSwitch"
python -m test.other.link_prediction --embedding_type="poincare_embeddings" --q=0.9 --dataset="ToggleSwitch"
python -m test.other.link_prediction --embedding_type="dmercator" --q=0.9 --dataset="ToggleSwitch"
python -m test.other.link_prediction --embedding_type="hydra" --q=0.9 --dataset="ToggleSwitch"
python -m test.other.link_prediction --embedding_type="hydra_plus" --q=0.9 --dataset="ToggleSwitch"
python -m test.other.link_prediction --embedding_type="hypermap" --q=0.9 --dataset="ToggleSwitch"
python -m test.other.link_prediction --embedding_type="lorentz" --q=0.9 --dataset="ToggleSwitch"


# python -m test.other.link_prediction --embedding_type="poincare_maps" --q=0.9 --dataset="Olsson"  --seed=42
# python -m test.other.link_prediction --embedding_type="poincare_embeddings" --q=0.9 --dataset="Olsson" --seed=42
# python -m test.other.link_prediction --embedding_type="dmercator" --q=0.9 --dataset="Olsson"  --seed=42
# python -m test.other.link_prediction --embedding_type="hydra" --q=0.9 --dataset="Olsson" --seed=42
# python -m test.other.link_prediction --embedding_type="hydra_plus" --q=0.9 --dataset="Olsson" --seed=42
# python -m test.other.link_prediction --embedding_type="hypermap" --q=0.9 --dataset="Olsson" --seed=42





# python -m test.other.link_prediction --embedding_type="poincare_maps" --q=0.9 --dataset="MyeloidProgenitors"
# python -m test.other.link_prediction --embedding_type="poincare_embeddings" --q=0.9 --dataset="MyeloidProgenitors"
# python -m test.other.link_prediction --embedding_type="dmercator" --q=0.9 --dataset="MyeloidProgenitors"
# python -m test.other.link_prediction --embedding_type="hydra" --q=0.9 --dataset="MyeloidProgenitors"
# python -m test.other.link_prediction --embedding_type="hydra_plus" --q=0.9 --dataset="MyeloidProgenitors"
# python -m test.other.link_prediction --embedding_type="hypermap" --q=0.9 --dataset="MyeloidProgenitors"


# python -m test.other.link_prediction --embedding_type="poincare_maps" --q=0.9 --dataset="krumsiek11_blobs"
# python -m test.other.link_prediction --embedding_type="poincare_embeddings" --q=0.9 --dataset="krumsiek11_blobs"
# python -m test.other.link_prediction --embedding_type="dmercator" --q=0.9 --dataset="krumsiek11_blobs"
# python -m test.other.link_prediction --embedding_type="hydra" --q=0.9 --dataset="krumsiek11_blobs"
# python -m test.other.link_prediction --embedding_type="hydra_plus" --q=0.9 --dataset="krumsiek11_blobs"
# python -m test.other.link_prediction --embedding_type="hypermap" --q=0.9 --dataset="krumsiek11_blobs"


# python -m test.other.link_prediction --embedding_type="poincare_maps" --q=0.9 --dataset="Paul"
# python -m test.other.link_prediction --embedding_type="poincare_embeddings" --q=0.9 --dataset="Paul"
# python -m test.other.link_prediction --embedding_type="dmercator" --q=0.9 --dataset="Paul"
# python -m test.other.link_prediction --embedding_type="hydra" --q=0.9 --dataset="Paul"  
# python -m test.other.link_prediction --embedding_type="hydra_plus" --q=0.9 --dataset="Paul"
# python -m test.other.link_prediction --embedding_type="hypermap" --q=0.9 --dataset="Paul"




python -m test.other.link_prediction --embedding_type="poincare_maps" --q=0.9 --dataset="ToggleSwitch" --n_runs=10 --seed=42
python -m test.other.link_prediction --embedding_type="poincare_maps" --q=0.9 --dataset="Olsson" --n_runs=10 --seed=42
python -m test.other.link_prediction --embedding_type="poincare_maps" --q=0.9 --dataset="MyeloidProgenitors" --n_runs=10 --seed=42



python -m test.other.link_prediction --embedding_type="rdpg" --q=0.9 --dataset="ToggleSwitch" --n_runs=10 --seed=42 --rdpg_dim=16


## Node classification

# print numero de clases y cantidad de nodos a predecir

python -m test.other.hyperbolic_knn \
    --dataset polblogs \
    --embedding_type "hydra_plus" \
    --n_iterations 3

python -m test.other.hyperbolic_knn \
    --dataset polblogs \
    --embedding_type "poincare_maps" \
    --n_iterations 3

python -m test.other.hyperbolic_knn \
    --dataset polblogs \
    --embedding_type "poincare_embeddings" \
    --n_iterations 3

python -m test.other.hyperbolic_knn \
    --dataset polblogs \
    --embedding_type "dmercator" \
    --n_iterations 3



python -m test.other.hyperbolic_knn \
    --dataset neuroseed \
    --embedding_type "hydra_plus" \
    --n_iterations 3





python -m test.other.hyperbolic_knn \
    --dataset neuroseed \
    --neuroseed_task edit_distance \
    --use_predefined_splits \
    --n_iterations 3 \
    --embedding_type hydra_plus


python -m test.other.hyperbolic_knn \
    --dataset neuroseed \
    --neuroseed_task edit_distance \
    --use_predefined_splits \
    --n_iterations 3 \
    --embedding_type poincare_maps \
    --k_neighbors 5 \
    --dim 2


python -m test.other.hyperbolic_knn \
    --dataset neuroseed \
    --neuroseed_task edit_distance \
    --use_predefined_splits \
    --n_iterations 3 \
    --embedding_type "poincare_maps" 

python -m test.other.hyperbolic_knn \
    --dataset neuroseed \
    --neuroseed_task edit_distance \
    --use_predefined_splits \
    --n_iterations 3 \
    --embedding_type "poincare_embeddings" 

python -m test.other.hyperbolic_knn \
    --dataset neuroseed \
    --neuroseed_task edit_distance \
    --use_predefined_splits \
    --n_iterations 3 \
    --embedding_type "dmercator" 



python -m test.other.hyperbolic_knn \
    --dataset polblogs \
    --embedding_type demercator \
    --n_iterations 3

