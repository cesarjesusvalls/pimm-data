"""
ND280StepReader — 3D energy-deposit segments from ND280 ``step/`` HDF5 files
(``format_version: 1``; schema: nd280-viewer ``docs/dataset-schema.md``).

Each ``event_XXX`` holds one row per ND280 ``TG4HitSegment``: Geant4 steps of
one track merged inside one sensitive volume. ``track_idx`` is the FK into
:class:`ND280LablReader`'s ``per_track`` table. Units: mm, ns, MeV.

Output dict:

    coord      (N,3) float32   segment midpoint
    energy     (N,1) float32   edep (MeV)
    time       (N,1) float32   start time (ns, Geant4 frame)
    track_idx  (N,)  int32     FK → labl.per_track
    subdet     (N,)  uint8     sub-detector code (ND::GeomId::Def)
    geom_id    (N,)  uint32    readout element holding the midpoint (0 = none)

    (include_physics=True adds:)
    start, end  (N,3) float32  segment end points
    energy_vis  (N,1) float32  visible (Birks-quenched) energy
    length      (N,1) float32  summed step length (mm)

The reader also exposes ``detector_geometry``: the drawing volumes
(``config/geometry``) plus ``baseline`` / ``geometry_hash`` of the first shard.
"""

import h5py
import numpy as np

from ._base import ShardReaderBase


class ND280StepReader(ShardReaderBase):
    """Reads 3D segments from ND280 ``step/`` files.

    Parameters
    ----------
    data_root, split, dataset_name : shard discovery
        (``{dataset_name}_step_*.h5``).
    min_segments : int
        Drop events with fewer than this many segments.
    include_physics : bool
        Also emit ``start``, ``end``, ``energy_vis``, ``length``.
    """

    _MODALITY = 'step'

    def __init__(self, data_root, split='', dataset_name='nd280',
                 min_segments=0, include_physics=True, **kwargs):
        self.data_root = data_root
        self.split = split
        self.dataset_name = dataset_name
        self.min_segments = int(min_segments)
        self.include_physics = bool(include_physics)
        self._init_shards()
        self.detector_geometry = self._read_detector_geometry()

    def _read_detector_geometry(self):
        geom = dict(baseline=None, geometry_hash=None, volumes=None)
        for h5_path in self.h5_files:
            try:
                f = h5py.File(h5_path, 'r', libver='latest', swmr=True)
            except OSError:
                continue  # dangling shard (F17)
            with f:
                cfg = f.get('config')
                if cfg is None:
                    break
                for k in ('baseline', 'geometry_hash'):
                    if k in cfg.attrs:
                        v = cfg.attrs[k]
                        geom[k] = v.decode() if isinstance(v, bytes) else str(v)
                if 'geometry' in cfg:
                    g = cfg['geometry']
                    geom['volumes'] = {
                        k[len('volume_'):]: (g[k].asstr()[:] if k == 'volume_name' else g[k][:])
                        for k in g}
            break
        return geom

    def _index_for_shard(self, h5_path):
        if self.min_segments > 0:
            with h5py.File(h5_path, 'r', libver='latest', swmr=True) as f:
                valid = [int(k.rsplit('_', 1)[1]) for k in f.keys()
                         if k.startswith('event_')
                         and int(f[k].attrs.get('n_segments', 0)) >= self.min_segments]
            return np.array(sorted(valid), dtype=np.int64)
        return super()._index_for_shard(h5_path)

    def read_event(self, idx):
        f, event_key = self._locate_event(idx)
        evt = f[event_key]
        start = np.stack([evt['start_x'][:], evt['start_y'][:], evt['start_z'][:]],
                         axis=1).astype(np.float32)
        end = np.stack([evt['end_x'][:], evt['end_y'][:], evt['end_z'][:]],
                       axis=1).astype(np.float32)
        data = {
            'coord': 0.5 * (start + end),
            'energy': evt['edep'][:].astype(np.float32)[:, None],
            'time': evt['start_t'][:].astype(np.float32)[:, None],
            'track_idx': evt['track_idx'][:].astype(np.int32),
            'subdet': evt['subdet'][:].astype(np.uint8),
            'geom_id': evt['geom_id'][:].astype(np.uint32),
        }
        if self.include_physics:
            data['start'] = start
            data['end'] = end
            data['energy_vis'] = evt['edep_vis'][:].astype(np.float32)[:, None]
            data['length'] = evt['length'][:].astype(np.float32)[:, None]
        return data
