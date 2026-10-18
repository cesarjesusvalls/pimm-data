"""
ND280SensorReader — calibrated sensor hits from ND280 ``sensor/`` HDF5 files
(``format_version: 1``; schema: nd280-viewer ``docs/dataset-schema.md``).

One row per recorded hit; a sensor can appear in several rows (TFB cycles,
SuperFGD pulses, TPC/HA-TPC waveform pulses). Every ND280 readout element
(bar end, SuperFGD fibre, TPC/HA-TPC pad drift column) is a row of the sensor
table stored once per file in ``config/sensor_*``.

Output dict:

    sensor_idx  (H,) int32    row of the sensor table
    Q           (H,) float32  calibrated charge (unit depends on sub-detector)
    T           (H,) float32  time (ns, Geant4 frame; TPC/HA-TPC include drift)
    sensor_table  dict        the file's sensor table (positions, half_size,
                              subdet, axis, view, module, end, geom_id)

    (include_waveforms=True adds the TPC/HA-TPC samples:)
    wave_offsets (H+1,) int32, wave_q / wave_t (S,) float32
"""

import h5py
import numpy as np

from ._base import ShardReaderBase

SENSOR_TABLE_KEYS = ('positions', 'half_size', 'subdet', 'axis', 'view',
                     'module', 'end', 'geom_id')


def read_sensor_table(h5_files):
    """Sensor table of the first shard that opens (shared across shards)."""
    for h5_path in h5_files:
        try:
            f = h5py.File(h5_path, 'r', libver='latest', swmr=True)
        except OSError:
            continue  # dangling shard (F17)
        with f:
            cfg = f.get('config')
            if cfg is None or 'sensor_positions' not in cfg:
                return None
            return {k: cfg[f'sensor_{k}'][:] for k in SENSOR_TABLE_KEYS
                    if f'sensor_{k}' in cfg}
    return None


class ND280SensorReader(ShardReaderBase):
    """Reads calibrated sensor hits from ND280 ``sensor/`` files.

    Parameters
    ----------
    data_root, split, dataset_name : shard discovery
        (``{dataset_name}_sensor_*.h5``).
    include_waveforms : bool
        Also emit the TPC/HA-TPC per-sample waveforms.
    """

    _MODALITY = 'sensor'

    def __init__(self, data_root, split='', dataset_name='nd280',
                 include_waveforms=False, **kwargs):
        self.data_root = data_root
        self.split = split
        self.dataset_name = dataset_name
        self.include_waveforms = bool(include_waveforms)
        self._init_shards()
        self.sensor_table = read_sensor_table(self.h5_files)

    def read_event(self, idx):
        f, event_key = self._locate_event(idx)
        evt = f[event_key]
        data = {
            'sensor_idx': evt['sensor_idx'][:].astype(np.int32),
            'Q': evt['Q'][:].astype(np.float32),
            'T': evt['T'][:].astype(np.float32),
            'sensor_table': self.sensor_table,
        }
        if self.include_waveforms and 'waveforms' in evt:
            w = evt['waveforms']
            data['wave_offsets'] = w['offsets'][:].astype(np.int32)
            data['wave_q'] = w['q'][:].astype(np.float32)
            data['wave_t'] = w['t'][:].astype(np.float32)
        return data
