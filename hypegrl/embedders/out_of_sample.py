"""
Out-of-sample extension for embedding new nodes without full refit.

Given existing embeddings X for nodes 1..N and a new node v connected
to some subset of existing nodes, find x_v by minimizing the loss
with respect to x_v only, holding X fixed.

This is the default strategy for node addition in gradient-based methods.
Edges from the new node to existing ones are treated as unknown if their
weights are not provided, and jointly optimized with x_v.
"""
# TODO: implement
