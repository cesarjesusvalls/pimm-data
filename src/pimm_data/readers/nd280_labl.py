"""
ND280LablReader — per-event label tables from ND280 ``labl/`` HDF5 files
(``format_version: 1``; schema: nd280-viewer ``docs/dataset-schema.md``).

Four scopes, emitted flat as ``labl_{scope}_{column}`` (the dataset layer
rebuilds the nested ``{event, interaction, particle, track}`` dict):

* ``per_event``       — ``t0`` (earliest interaction time)
* ``per_interaction`` — one row per generator vertex: ``source_type``,
  ``generator``, ``reaction``, ``reaction_code``, ``t0``, ``vertex_{x,y,z}``,
  ``vertex_subdet``, ``neutrino_pdg``, ``neutrino_energy_MeV``,
  ``target_pdg``, ``n_primaries``, ``n_particles`` and CSR
  ``primary_{track_ids,pdgs,energies}_{offsets,data}``
* ``per_particle``    — one row per Geant4 primary: ``pdg`` (the v0 label),
  ``track_id``, ``interaction_idx``, ``initial_energy``,
  ``initial_momentum_{x,y,z}``, ``start_{x,y,z,t}``, ``edep``,
  ``n_segments``, ``n_hits``
* ``per_track``       — one row per saved Geant4 trajectory: ``track_id``,
  ``parent_id``, ``pdg``, ``initial_energy``, ``initial_momentum_{x,y,z}``,
  ``start_{x,y,z,t}``, ``end_{x,y,z,t}``, ``particle_idx``, ``ancestor``,
  ``interaction``

String columns (``generator``, ``reaction``) are returned as ``str`` object
arrays; everything else keeps its on-disk dtype.
"""

import numpy as np

from ._base import ShardReaderBase

_SCOPES = (('per_event', 'event'), ('per_interaction', 'interaction'),
           ('per_particle', 'particle'), ('per_track', 'track'))


class ND280LablReader(ShardReaderBase):
    """Reads per-event label tables from ND280 ``labl/`` files."""

    _MODALITY = 'labl'

    def __init__(self, data_root, split='', dataset_name='nd280', **kwargs):
        self.data_root = data_root
        self.split = split
        self.dataset_name = dataset_name
        self._init_shards()

    def read_event(self, idx):
        f, event_key = self._locate_event(idx)
        evt = f[event_key]
        data = {}
        for group, scope in _SCOPES:
            if group not in evt:
                continue
            g = evt[group]
            for k in g:
                ds = g[k]
                if ds.dtype.kind in ('O', 'S'):
                    v = np.array(ds.asstr()[()], dtype=object)
                else:
                    v = ds[()]
                data[f'labl_{scope}_{k}'] = v
        return data
