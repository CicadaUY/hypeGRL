"""
Efficient incremental updates for the forest matrix Q = (I + L)^{-1}.

Edge updates (rank-2 Woodbury)
-------------------------------
Adding or removing edge (m, n) changes L by +/- B_mn where
B_mn = (e_m - e_n)(e_m - e_n)^T. The forest matrix updates as::

    Q_new = Q - Q @ B_mn @ Q / (1 + tr(B_mn @ Q))

which decomposes into two rank-1 Sherman-Morrison steps, each O(N^2).

Node addition (bordered matrix inverse)
----------------------------------------
Adding node v with edges to existing nodes changes (I + L) to a
bordered matrix. Its inverse is computed via the Schur complement::

    s   = c - b^T @ Q @ b
    Q_new = block([[Q + Q@b@b^T@Q/s,  -Q@b/s ],
                   [      -b^T@Q/s,     1/s  ]])

where b encodes the edges from v to existing nodes and c = 1 + deg(v).
This is also O(N^2).

Node deletion
-------------
Removing node v corresponds to removing the v-th row and column from
(I + L) and recomputing the inverse of the resulting (N-1)x(N-1) matrix.
Handled via the inverse of a principal submatrix using the
matrix inversion lemma. O(N^2).

Connectivity check
------------------
All update functions check whether the resulting graph remains connected
and raise ValueError if not, since a disconnected graph produces a
block-diagonal L whose forest matrix changes character.
"""
# TODO: implement
