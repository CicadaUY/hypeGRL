"""Reproduction scripts for the papers built on hypeGRL.

One subpackage per paper (``hypegrl_paper``, ``icassp2027``) plus ``exploratory``,
over shared dataset loaders and per-graph descriptors. This package is *not*
installed with the library — it
carries the heavy, niche dependencies (RDPG/ASE baselines, dataset formats)
and has no API-stability contract. It builds on the reusable primitives in
:mod:`hypegrl.evaluation`.
"""
