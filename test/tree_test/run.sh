python -m test.tree_test.main --embedding_type='poincare_maps' --output_space='poincare'
python -m test.tree_test.main --embedding_type='poincare_maps' --output_space='spherical'

python -m test.tree_test.link_prediction --embedding_type="poincare_maps" --q=0.9 --n_links=20
python -m test.tree_test.link_prediction --embedding_type="poincare_embeddings" --q=0.9 --n_links=20
python -m test.tree_test.link_prediction --embedding_type="hydra" --q=0.9 --n_links=20
python -m test.tree_test.link_prediction --embedding_type="hydra_plus" --q=0.9 --n_links=20



python -m test.tree_test.link_prediction --dataset="ToggleSwitch" --embedding_type="rdpg" --rdpg_dim 4 --q=0.9 --seed=42  --num_runs=10
python -m test.tree_test.link_prediction --dataset="ToggleSwitch" --embedding_type rdpg --rdpg_dim 8 --q=0.9 --seed=42  --num_runs=10


python -m test.other.link_prediction --dataset="ToggleSwitch" --embedding_type="rdpg" --rdpg_dim=4 --q=0.9 --seed=42  --num_runs=10
