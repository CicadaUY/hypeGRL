# python -m test.karate_club.main --embedding_type='poincare_maps' --output_space='poincare'
# python -m test.karate_club.main --embedding_type='poincare_maps' --output_space='spherical'

python -m test.karate_club.link_prediction --embedding_type="poincare_maps" --q=0.9
python -m test.karate_club.link_prediction --embedding_type="poincare_embeddings" --q=0.9
python -m test.karate_club.link_prediction --embedding_type="hydra" --q=0.9
python -m test.karate_club.link_prediction --embedding_type="hydra_plus" --q=0.9
python -m test.karate_club.link_prediction --embedding_type="dmercator" --q=0.9
python -m test.karate_club.link_prediction --embedding_type="lorentz" --q=0.9
python -m test.karate_club.link_prediction --embedding_type="hypermap" --q=0.9