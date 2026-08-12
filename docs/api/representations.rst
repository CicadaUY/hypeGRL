Representations
===============

A ``Representation`` is the *chart* an embedding is stored and optimised in —
the coordinates the optimiser actually moves. It is orthogonal to the method:
every gradient embedder takes a ``representation=`` argument selecting one,
and the loss only ever sees ``rep.dist()``, the exact pairwise hyperbolic
distance.

Charts are not interchangeable in practice, because real graphs place
low-degree nodes at large radius, where the ambient coordinates lose
resolution. The polar chart keeps the radius a plain number and stays exact
there; see each class below for its range.

The base class
--------------

.. automodule:: hypegrl.representations.base
   :members:
   :show-inheritance:

Polar charts
------------

.. automodule:: hypegrl.representations.polar
   :members:
   :show-inheritance:

Ambient charts
--------------

.. automodule:: hypegrl.representations.ball
   :members:
   :show-inheritance:

.. automodule:: hypegrl.representations.hyperboloid
   :members:
   :show-inheritance:

Tangent chart
-------------

.. automodule:: hypegrl.representations.tangent
   :members:
   :show-inheritance:
