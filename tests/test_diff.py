import numpy as np

from watcher.diff import FrameComparer, hellinger


def _frame(fill, h=200, w=320):
    return np.full((h, w, 4), fill, dtype=np.uint8)


def test_first_frame_is_full_change():
    c = FrameComparer()
    assert c.compare(_frame(0)) == 1.0


def test_identical_frame_scores_zero():
    c = FrameComparer()
    c.compare(_frame(120))
    assert c.compare(_frame(120)) == 0.0  # bit-identical => hash early-exit


def test_big_change_scores_high():
    c = FrameComparer()
    c.compare(_frame(0))       # all black
    score = c.compare(_frame(255))  # all white
    assert score > 0.9


def test_small_change_scores_low():
    c = FrameComparer()
    base = _frame(128)
    c.compare(base)
    changed = base.copy()
    # flip a tiny patch to a clearly different intensity (crosses a histogram bin)
    # -> a real but minimal shift, like a small on-screen element changing.
    changed[0:4, 0:4, :3] = 200
    score = c.compare(changed)
    assert 0.0 < score < 0.05


def test_hellinger_bounds():
    p = np.array([1.0, 0.0])
    q = np.array([0.0, 1.0])
    assert abs(hellinger(p, p)) < 1e-9
    assert abs(hellinger(p, q) - 1.0) < 1e-9
