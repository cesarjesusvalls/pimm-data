"""Tests for ND280Dataset (nd280-viewer docs/dataset-schema.md, format_version 1).

Synthetic fixtures by default; set ND280_DATA_ROOT to a produced dataset root
to run the same checks on real ND280 output (original or upgraded geometry)."""

import numpy as np
import pytest

from pimm_data import ND280Dataset

SFG, TPC = 9, 2


def make_ds(nd280_data_root, **kwargs):
    defaults = dict(data_root=nd280_data_root, split='', dataset_name='nd280')
    defaults.update(kwargs)
    return ND280Dataset(**defaults)


def test_sensor_only(nd280_data_root):
    ds = make_ds(nd280_data_root, modalities=('sensor',))
    s = ds.get_data(0)['sensor']
    n = s['coord'].shape[0]
    assert s['coord'].shape == (n, 3) and s['half_size'].shape == (n, 3)
    assert s['energy'].shape == (n, 1) and s['time'].shape == (n, 1)
    assert s['sensor_idx'].shape == s['subdet'].shape == s['axis'].shape == (n,)
    assert 'segment' not in s and 'instance' not in s
    # coord is the readout element's centre from the file's sensor table
    table = ds.sensor_reader.sensor_table
    assert np.allclose(s['coord'], table['positions'][s['sensor_idx']])


def test_step_only(nd280_data_root):
    ds = make_ds(nd280_data_root, modalities=('step',))
    st = ds.get_data(0)['step']
    n = st['coord'].shape[0]
    assert st['energy'].shape == (n, 1) and st['time'].shape == (n, 1)
    for k in ('track_idx', 'subdet', 'geom_id'):
        assert st[k].shape == (n,)
    for k in ('start', 'end'):
        assert st[k].shape == (n, 3)
    assert np.allclose(st['coord'], 0.5 * (st['start'] + st['end']))
    assert 'segment' not in st


def test_step_include_physics_false(nd280_data_root):
    st = make_ds(nd280_data_root, modalities=('step',), include_physics=False).get_data(0)['step']
    assert 'start' not in st and 'energy_vis' not in st


def test_hits_only(nd280_data_root):
    h = make_ds(nd280_data_root, modalities=('hits',)).get_data(0)['hits']
    assert np.array_equal(h['instance'], h['particle_idx'])
    assert 'segment' not in h


@pytest.mark.parametrize('mods', [(), ('labl',), ('sensor', 'labl'), ('nope',)])
def test_invalid_modalities(nd280_data_root, mods):
    with pytest.raises(ValueError):
        make_ds(nd280_data_root, modalities=mods)


def test_hits_shares_sum_to_sensor_charge(nd280_data_root):
    ds = make_ds(nd280_data_root, modalities=('sensor', 'hits'))
    for i in range(len(ds)):
        d = ds.get_data(i)
        q = d['sensor']['energy'][:, 0]
        per_digit = np.bincount(d['hits']['digit_idx'], weights=d['hits']['energy'][:, 0],
                                minlength=len(q))
        assert np.allclose(per_digit, q, rtol=1e-4, atol=1e-3)
        # hits rows point at the sensor of their digit
        assert np.array_equal(d['hits']['sensor_idx'], d['sensor']['sensor_idx'][d['hits']['digit_idx']])


def test_hits_plus_labl_labels(nd280_data_root):
    ds = make_ds(nd280_data_root, modalities=('hits', 'labl'), ignore_index=-1)
    d = ds.get_data(0)
    h, pdg = d['hits'], d['labl']['particle']['pdg']
    real = h['particle_idx'] >= 0
    assert np.array_equal(h['segment'][real], pdg[h['particle_idx'][real]])
    noise = h['contribution_type'] == 2
    assert np.all(h['particle_idx'][noise] == -1) and np.all(h['segment'][noise] == -1)


def test_step_plus_labl_labels(nd280_data_root):
    d = make_ds(nd280_data_root, modalities=('step', 'labl')).get_data(0)
    st, lb = d['step'], d['labl']
    expected = lb['track']['particle_idx'][st['track_idx']]
    assert np.array_equal(st['particle_idx'], expected)
    assert np.array_equal(st['instance'], expected)
    assert np.array_equal(st['segment'], lb['particle']['pdg'][expected])


def test_all_four_modalities(nd280_data_root):
    ds = make_ds(nd280_data_root, modalities=('step', 'sensor', 'hits', 'labl'))
    for i in range(len(ds)):
        d = ds.get_data(i)
        assert {'step', 'sensor', 'hits', 'labl'} <= set(d)
        n_part = d['labl']['particle']['pdg'].shape[0]
        assert d['hits']['particle_idx'].max() < n_part
        assert d['step']['particle_idx'].max() < n_part
        assert set(d['labl']) == {'event', 'interaction', 'particle', 'track'}


def test_subdet_filter(nd280_data_root):
    d = make_ds(nd280_data_root, modalities=('step', 'sensor', 'hits', 'labl'),
                subdets=(SFG,)).get_data(0)
    for m in ('step', 'sensor', 'hits'):
        assert np.all(d[m]['subdet'] == SFG)


def test_include_noise_false(nd280_data_root):
    h = make_ds(nd280_data_root, modalities=('hits',), include_noise=False).get_data(0)['hits']
    assert np.all(h['contribution_type'] != 2) and np.all(h['particle_idx'] >= 0)


def test_include_waveforms(nd280_data_root):
    ds = make_ds(nd280_data_root, modalities=('sensor',), include_waveforms=True)
    s = ds.get_data(0)['sensor']
    off = s['wave_offsets']
    assert off.shape[0] == s['coord'].shape[0] + 1
    assert off[-1] == s['wave_q'].shape[0] == s['wave_t'].shape[0]
    # waveforms only on TPC / HA-TPC pads
    has_wave = np.diff(off) > 0
    assert np.all(np.isin(s['subdet'][has_wave], (TPC, 10)))
    # and survive a sub-detector filter consistently
    s2 = make_ds(nd280_data_root, modalities=('sensor',), include_waveforms=True,
                 subdets=(TPC,)).get_data(0)['sensor']
    assert s2['wave_offsets'][-1] == s2['wave_q'].shape[0]
    assert np.all(s2['subdet'] == TPC)


def test_detector_geometry(nd280_data_root):
    ds = make_ds(nd280_data_root, modalities=('step', 'sensor'))
    g = ds.detector_geometry
    assert g['baseline'] is not None and g['geometry_hash'] is not None
    v = g['volumes']
    assert v['center'].shape == v['half_size'].shape and v['center'].shape[1] == 3
    assert len(v['name']) == v['center'].shape[0]
    assert g['sensor_table']['positions'].shape[1] == 3


def test_len_and_getitem(nd280_data_root):
    ds = make_ds(nd280_data_root, modalities=('sensor', 'hits', 'labl'))
    assert len(ds) > 0
    sample = ds[0]
    assert 'sensor' in sample and 'hits' in sample


def test_min_segments_requires_step(nd280_data_root):
    with pytest.raises(ValueError):
        make_ds(nd280_data_root, modalities=('sensor',), min_segments=5)


@pytest.mark.real_data_only
def test_real_data_invariants(nd280_data_root):
    """On produced data: shares add up and every segment has a particle."""
    ds = make_ds(nd280_data_root, modalities=('step', 'sensor', 'hits', 'labl'))
    for i in range(min(len(ds), 20)):
        d = ds.get_data(i)
        q = d['sensor']['energy'][:, 0]
        per_digit = np.bincount(d['hits']['digit_idx'], weights=d['hits']['energy'][:, 0],
                                minlength=len(q))
        assert np.allclose(per_digit, q, rtol=1e-4, atol=1e-2)
        assert np.all(d['step']['particle_idx'] >= 0)
