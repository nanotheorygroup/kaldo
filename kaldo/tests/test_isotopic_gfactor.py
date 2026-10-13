"""
Unit and regression test for the kaldo package.

Covers the isotopic g factor lookup and its offline fallback. The NIST table is fetched over the
network, which is routinely unavailable on HPC compute nodes, so the bundled legacy database has to
take over cleanly. The network is the one thing that cannot be exercised in a test, so the HTTP call
is monkeypatched to reproduce each failure mode it can raise; a single opt-in test below talks to
NIST for real and skips when it cannot.
"""

import json
import socket
import urllib.error
from importlib import resources as impresources

import numpy as np
import pytest

import kaldo.controllers
from kaldo.controllers import isotopic


class FakeResponse:
    """Stands in for the object urlopen returns as a context manager."""

    def __init__(self, payload):
        self.payload = payload

    def read(self):
        return self.payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


@pytest.fixture
def legacy_gfactors():
    dataset_file = impresources.files(kaldo.controllers) / 'legacy_dataset.json'
    with dataset_file.open('r') as file:
        return json.load(file)


@pytest.fixture
def captured_request(monkeypatch):
    """Capture the outgoing request without touching the network."""
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured['request'] = req
        captured['timeout'] = timeout
        return FakeResponse(b'ignored, the parser is stubbed out')

    monkeypatch.setattr(isotopic.request, 'urlopen', fake_urlopen)
    monkeypatch.setattr(isotopic, 'parse_isotope_data', lambda raw: two_isotope_table(14))
    return captured


@pytest.fixture
def offline(monkeypatch):
    """Make the download fail the way an unreachable network does."""

    def fail(*args, **kwargs):
        raise urllib.error.URLError('Name or service not known')

    monkeypatch.setattr(isotopic.request, 'urlopen', fail)


def two_isotope_table(atomic_number):
    # Equal parts mass 1 and mass 3 gives an average mass of 2 and g = 0.5 * 0.5^2 * 2 = 0.25.
    return {atomic_number: {1: {'mass': 1.0, 'composition': 0.5}, 3: {'mass': 3.0, 'composition': 0.5}}}


def raises(error):
    def fail(*args, **kwargs):
        raise error

    return fail


# ---------------------------------------------------------------------------------------------
# the download itself
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize('error', [
    urllib.error.URLError('no route to host'),
    urllib.error.HTTPError('http://nist.gov', 403, 'Forbidden', {}, None),
    TimeoutError('timed out'),
    socket.gaierror('name resolution failed'),
    ValueError('captive portal served html instead of the NIST table'),
    IndexError('truncated response'),
    UnicodeDecodeError('utf-8', b'\xff', 0, 1, 'invalid start byte'),
])
def test_download_returns_none_instead_of_raising(monkeypatch, error):
    monkeypatch.setattr(isotopic.request, 'urlopen', raises(error))
    assert isotopic.download_isotopes() is None


def test_download_does_not_swallow_keyboard_interrupt(monkeypatch):
    monkeypatch.setattr(isotopic.request, 'urlopen', raises(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        isotopic.download_isotopes()


def test_download_rejects_a_response_that_parses_to_nothing(monkeypatch):
    # An error page or captive portal parses without raising, it simply yields no elements.
    monkeypatch.setattr(isotopic.request, 'urlopen', lambda req, timeout=None: FakeResponse(b'<html>bot check</html>'))
    monkeypatch.setattr(isotopic, 'parse_isotope_data', lambda raw: {})
    assert isotopic.download_isotopes() is None


def test_download_identifies_itself_as_kaldo(captured_request):
    # NIST sits behind Cloudflare, which 403s the default 'Python-urllib' agent. Sending an agent
    # that names the project is what keeps the online path working.
    isotopic.download_isotopes()
    user_agent = captured_request['request'].get_header('User-agent')
    assert 'kaldo' in user_agent
    assert 'Python-urllib' not in user_agent


def test_download_requests_the_nist_table_over_https(captured_request):
    isotopic.download_isotopes()
    url = captured_request['request'].full_url
    assert url.startswith('https://')
    assert 'physics.nist.gov' in url


def test_download_applies_a_timeout(captured_request):
    # Without this a firewall that blackholes packets hangs the calculation for minutes.
    isotopic.download_isotopes(timeout=7)
    assert captured_request['timeout'] == 7


def test_download_timeout_defaults_to_the_module_constant(captured_request):
    isotopic.download_isotopes()
    assert captured_request['timeout'] == isotopic.ISOTOPE_DOWNLOAD_TIMEOUT


# ---------------------------------------------------------------------------------------------
# choosing between online and legacy data
# ---------------------------------------------------------------------------------------------


def test_gfactor_falls_back_to_legacy_data_when_offline(offline, legacy_gfactors):
    g_factor = isotopic.compute_gfactor(np.array([14, 14]))
    np.testing.assert_allclose(g_factor, np.full(2, legacy_gfactors['14']), rtol=1e-12, atol=0.0)


def test_gfactor_maps_each_atom_to_its_own_element(offline, legacy_gfactors):
    g_factor = isotopic.compute_gfactor(np.array([12, 8, 12, 8]))
    expected = np.array([legacy_gfactors['12'], legacy_gfactors['8'], legacy_gfactors['12'], legacy_gfactors['8']])
    np.testing.assert_allclose(g_factor, expected, rtol=1e-12, atol=0.0)


def test_gfactor_uses_downloaded_data_when_available(monkeypatch):
    monkeypatch.setattr(isotopic, 'download_isotopes', lambda: two_isotope_table(14))
    np.testing.assert_allclose(isotopic.compute_gfactor(np.array([14])), [0.25], rtol=1e-12, atol=0.0)


def test_gfactor_falls_back_when_downloaded_data_misses_an_element(monkeypatch, legacy_gfactors):
    # A truncated or partially parsed response must not produce a KeyError halfway through the loop.
    monkeypatch.setattr(isotopic, 'download_isotopes', lambda: two_isotope_table(14))
    g_factor = isotopic.compute_gfactor(np.array([14, 32]))
    expected = np.array([legacy_gfactors['14'], legacy_gfactors['32']])
    np.testing.assert_allclose(g_factor, expected, rtol=1e-12, atol=0.0)


# ---------------------------------------------------------------------------------------------
# elements without a natural isotopic composition
# ---------------------------------------------------------------------------------------------


def test_unstable_element_raises_instead_of_returning_nan(offline):
    # Technetium (Z=43) has no stable isotopic composition; the legacy database stores it as null.
    with pytest.raises(ValueError, match='Tc'):
        isotopic.compute_gfactor(np.array([43]))


def test_unstable_element_from_downloaded_data_raises(monkeypatch):
    # NIST lists elements without natural abundance at zero composition, which divides to nan.
    table = {43: {97: {'mass': 96.9, 'composition': 0.0}, 99: {'mass': 98.9, 'composition': 0.0}}}
    monkeypatch.setattr(isotopic, 'download_isotopes', lambda: table)
    with pytest.raises(ValueError, match='Tc'):
        isotopic.compute_gfactor(np.array([43]))


def test_every_stable_element_has_a_finite_legacy_gfactor(offline, legacy_gfactors):
    stable = [int(z) for z, g in legacy_gfactors.items() if g is not None]
    g_factor = isotopic.compute_gfactor(np.array(stable))
    assert np.all(np.isfinite(g_factor))
    assert np.all(g_factor >= 0.0)


# ---------------------------------------------------------------------------------------------
# live network
# ---------------------------------------------------------------------------------------------


def test_live_nist_download_agrees_with_the_legacy_database(legacy_gfactors):
    """Guards against NIST moving the table or tightening its bot rules again.

    Skips rather than fails when the machine has no route to NIST, which is the normal state on a
    compute node and the whole reason the legacy database exists.
    """
    isotopes = isotopic.download_isotopes()
    if isotopes is None:
        pytest.skip('NIST is not reachable from this machine')

    stable = [int(z) for z, g in legacy_gfactors.items() if g is not None]
    online = isotopic.compute_gfactor(np.array(stable))
    expected = np.array([legacy_gfactors[str(z)] for z in stable])
    np.testing.assert_allclose(online, expected, rtol=1e-6, atol=1e-12)
