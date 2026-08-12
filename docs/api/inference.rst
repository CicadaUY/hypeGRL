Inference
=========

The optimisers every gradient-based embedder delegates to, the machinery for
partially observed graphs, and the estimators for the scalar parameters of the
latent-geometry model.

Both optimisers consume a
:class:`~hypegrl.representations.base.Representation` and a loss closure
``loss_fn(rep, A)``; an embedder picks between them according to whether any
edge is unknown.

Riemannian optimiser (fixed target)
-----------------------------------

.. automodule:: hypegrl.inference.riemannian_optimizer
   :members:

Joint optimiser (unknown edges)
-------------------------------

.. automodule:: hypegrl.inference.joint_optimizer
   :members:

Imputation
----------

.. automodule:: hypegrl.inference.imputation
   :members:

Parameter estimation
--------------------

.. automodule:: hypegrl.inference.parameters
   :members:
