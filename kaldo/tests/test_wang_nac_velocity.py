import numpy as np
import pytest

from kaldo.observables.harmonic_with_q import HarmonicWithQ
from kaldo.tests.test_wang_nac_api import nac_second_order
from kaldo.tests.wang_debug_reference import (
    format_tensor_diff,
    load_q_tensor,
    require_wang_att3_debug,
)

_VELOCITY_Q_NAMES = ["q-00000", "q-00010", "q-00020"]
_VELOCITY_RTOL = 0.1
_VELOCITY_ATOL = 7.0


@pytest.mark.parametrize("q_name", _VELOCITY_Q_NAMES)
def test_wang_group_velocities_match_phonopy_debug(nac_second_order, q_name):
    debug_dir = require_wang_att3_debug()
    # Phonopy group velocities are computed at py_derivative_q (the displaced q),
    # not at py_qpoints. py_derivative_q_direction is always [0,0,0] for these points.
    q_red = load_q_tensor(debug_dir, q_name, "py_derivative_q").flatten()
    phonon = HarmonicWithQ(
        q_point=q_red,
        second=nac_second_order,
        storage="memory",
        is_unfolding=True,
        nac_method="wang",
        q_index=int(q_name.split("-")[1]),
    )
    actual = phonon.velocity[0]
    expected = load_q_tensor(debug_dir, q_name, "py_group_velocities")
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=_VELOCITY_RTOL,
        atol=_VELOCITY_ATOL,
        err_msg=format_tensor_diff("group_velocities", q_name, actual, expected),
    )
