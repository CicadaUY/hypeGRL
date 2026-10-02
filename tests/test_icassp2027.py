"""Tests for the ICASSP 2027 manifest: the inputs and invariants behind its numbers."""
import numpy as np
import pytest

from experiments.icassp2027.make_table import collect
from experiments.icassp2027.manifest import FLAT, GRAPHS, best, final_stress, runs
from experiments.icassp2027.reference import Reference, plotted_runs
from experiments.icassp2027.two_stage_chart_schedule import load_graph


@pytest.mark.parametrize("name", list(GRAPHS))
def test_every_graph_loads_as_a_connected_graph(name):
    G = load_graph(GRAPHS[name].graph)
    assert G.number_of_nodes() >= 200
    assert set(G.nodes()) == set(range(G.number_of_nodes()))


def test_every_run_writes_its_own_files():
    """Two runs sharing a stem would silently overwrite each other's results."""
    stems = [r.stem("cuda") for r in runs()]
    assert len(stems) == len(set(stems))


def test_each_handover_exists_and_runs_before_the_blocks_using_it():
    order = runs()
    position = {(r.name, r.kind, r.curvature, r.rates): i for i, r in enumerate(order)}
    for i, r in enumerate(order):
        if r.kind == "stage2":
            h = r.handover()
            assert h.rates[0] in GRAPHS[r.name].stage1[1]
            assert position[(h.name, h.kind, h.curvature, h.rates)] < i


def test_single_curvature_budgets_match_the_two_stage_budget_except_cichlidae():
    """The schedule gain compares equal budgets; the only exceptions are the three
    half-budget cichlidae runs the paper reports with a dagger."""
    short = set()
    for name, g in GRAPHS.items():
        total = g.stage1[0] + g.stage2[0]
        short |= {(name, c) for c, (n, _) in g.single.items() if n != total}
    assert short == {("cichlidae", c) for c in ("c=0.2", "c=0.3", "c=1.0")}


def test_both_stages_and_all_four_curvatures_are_covered():
    for g in GRAPHS.values():
        assert g.stage1_lr in g.stage1[1]
        assert set(g.stage2[1]) == {FLAT, "c=0.2", "c=0.3"}
        assert set(g.single) == {FLAT, "c=0.2", "c=0.3", "c=1.0"}


def test_final_stress_is_the_mean_of_the_last_tenth():
    curve = np.r_[np.full(90, 100.0), np.full(10, 2.0)]
    assert final_stress(curve) == 2.0
    assert final_stress(np.r_[curve, np.inf]) == float("inf")


def test_best_skips_diverged_rates_and_says_why_when_it_cannot_choose():
    assert best({0.1: 3.0, 0.3: 5.0, 1.0: float("inf")}) == (0.1, 3.0)
    with pytest.raises(ValueError, match="diverged"):
        best({1.0: float("inf")})
    with pytest.raises(ValueError, match="no finished runs"):
        best({})


def test_the_reference_holds_exactly_one_row_per_arm_of_the_manifest():
    ref = Reference()
    arms = {(r.name, r.kind, r.curvature, r.iterations, lr)
            for r in runs() for lr in r.rates}
    assert set(ref.rows) == arms


def test_the_reference_keeps_every_curve_the_figures_draw():
    ref = Reference()
    for name in GRAPHS:
        for group in plotted_runs(name):
            scores = {}
            for run in group:
                scores.update({lr: (run, s) for lr, s in ref.scores(run).items()})
            lr, _ = best({lr: s for lr, (_, s) in scores.items()})
            x, y = ref.points(scores[lr][0], lr)
            assert len(x) == len(y) > 0 and np.isfinite(y).all()


@pytest.mark.parametrize("name, curvature_gain, schedule_gain", [
    ("caterpillar", -98.7, -98.0), ("fabaceae_sub", -26.5, -60.0),
    ("cichlidae", -23.1, -45.8), ("powerlaw200", 0.0, 0.4)])
def test_the_reference_reproduces_the_published_gains(name, curvature_gain,
                                                       schedule_gain):
    """The committed numbers are the paper's; a reference regenerated from the
    wrong runs would change them."""
    res = collect(name, Reference())
    assert round(res["curvature_gain"], 1) == pytest.approx(curvature_gain)
    assert round(res["schedule_gain"], 1) == pytest.approx(schedule_gain)
