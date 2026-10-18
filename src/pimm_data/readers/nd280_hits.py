"""
ND280HitsReader — per-particle decomposition of ND280 sensor hits from
``hits/`` HDF5 files (``format_version: 1``; schema: nd280-viewer
``docs/dataset-schema.md``).

Each row is one ``(digit_idx, particle_idx)`` contribution: a particle's share
of one recorded sensor hit. Summing ``Q`` over the rows of a ``digit_idx``
reproduces the ``sensor/`` charge. Noise hits carry ``particle_idx = -1`` and
``contribution_type = 2``.

Output dict:

    particle_idx       (E,) int32    FK → labl.per_particle; -1 = noise
    digit_idx          (E,) int32    FK → sensor row of the event
    sensor_idx         (E,) int32    row of the sensor table
    Q                  (E,) float32  the particle's share of the hit charge
    T_true             (E,) float32  earliest Geant4 time of its deposits (ns)
    contribution_type  (E,) int8     0 direct, 1 indirect (crosstalk,
                                     diffusion, charge spreading), 2 noise
    sensor_table       dict          the file's sensor table
"""

import numpy as np

from ._base import ShardReaderBase
from .nd280_sensor import read_sensor_table

NOISE = 2


class ND280HitsReader(ShardReaderBase):
    """Reads per-particle hit decomposition from ND280 ``hits/`` files.

    Parameters
    ----------
    data_root, split, dataset_name : shard discovery
        (``{dataset_name}_hits_*.h5``).
    include_noise : bool
        Keep the noise rows (``contribution_type == 2``).
    """

    _MODALITY = 'hits'

    def __init__(self, data_root, split='', dataset_name='nd280',
                 include_noise=True, **kwargs):
        self.data_root = data_root
        self.split = split
        self.dataset_name = dataset_name
        self.include_noise = bool(include_noise)
        self._init_shards()
        self.sensor_table = read_sensor_table(self.h5_files)

    def read_event(self, idx):
        f, event_key = self._locate_event(idx)
        evt = f[event_key]
        data = {
            'particle_idx': evt['particle_idx'][:].astype(np.int32),
            'digit_idx': evt['digit_idx'][:].astype(np.int32),
            'sensor_idx': evt['sensor_idx'][:].astype(np.int32),
            'Q': evt['Q'][:].astype(np.float32),
            'T_true': evt['T_true'][:].astype(np.float32),
            'contribution_type': evt['contribution_type'][:].astype(np.int8),
        }
        if not self.include_noise:
            keep = data['contribution_type'] != NOISE
            data = {k: v[keep] for k, v in data.items()}
        data['sensor_table'] = self.sensor_table
        return data
