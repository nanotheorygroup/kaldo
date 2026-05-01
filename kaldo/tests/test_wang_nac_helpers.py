from kaldo.tests.wang_debug_reference import (
    diagnostic_q_names_wang_att3,
    require_wang_att3_debug,
)


def test_wang_att3_debug_tree_is_loadable():
    root = require_wang_att3_debug()
    assert (root / "q-00000" / "py_qpoints.npy").exists()


def test_wang_diagnostic_q_names_match_expected_points():
    assert diagnostic_q_names_wang_att3() == [
        "q-00000",
        "q-00010",
        "q-00020",
        "q-00030",
    ]
