"""
ND280Dataset — multimodal dataset for labelled T2K ND280 simulation
(``format_version: 1``; schema and producer: nd280-viewer
``docs/dataset-schema.md``). Works for the original (e.g. baseline
``2010-11``) and the upgraded (``2024``: SuperFGD, HA-TPCs, ToF) detector: all
geometry travels with the data.

Loads from co-indexed per-modality HDF5 shards:

* ``step/``   — 3D Geant4 energy-deposit segments (ND280 ``TG4HitSegment``)
* ``sensor/`` — calibrated sensor hits of every sub-detector (2D readout)
* ``hits/``   — per-particle decomposition of the sensor hits
* ``labl/``   — per-event / per-interaction / per-particle / per-track tables

Returns a **nested** dict (units mm, ns, MeV)::

    {
      'step':   {'coord': (N,3), 'energy': (N,1), 'time': (N,1),
                 'track_idx': (N,), 'subdet': (N,), 'geom_id': (N,),
                 'start', 'end', 'energy_vis', 'length',      # include_physics
                 'particle_idx', 'instance', 'segment'},      # with labels
      'sensor': {'coord': (H,3), 'energy': (H,1), 'time': (H,1),
                 'sensor_idx': (H,), 'subdet': (H,), 'axis': (H,),
                 'half_size': (H,3), 'wave_*'},               # include_waveforms
      'hits':   {'coord': (E,3), 'energy': (E,1), 'time': (E,1),
                 'sensor_idx', 'digit_idx', 'subdet', 'contribution_type',
                 'particle_idx', 'instance', 'segment'},
      'labl':   {'event': {...}, 'interaction': {...}, 'particle': {...},
                 'track': {...}},
      'name': str, 'split': str,
    }

``coord`` of a sensor row is the centre of its readout element (bar, fibre,
or TPC/HA-TPC pad drift column); ``axis`` is the element's long, unmeasured
axis (0=x, 1=y, 2=z) and ``half_size`` its half-extents.

Labels (v0): a *particle* is a Geant4 primary; ``instance = particle_idx`` and
``segment = labl.particle.pdg`` (raw PDG — remap downstream). Noise hits have
``particle_idx = -1`` and ``segment = ignore_index``.

Registered in :data:`pimm_data.DATASETS`.
"""

import numpy as np

from .builder import DATASETS
from ._dataset_base import ShardEventDataset
from ._label_decorate import decorate_labels, gather_with_fill
from .readers.nd280_step import ND280StepReader
from .readers.nd280_sensor import ND280SensorReader
from .readers.nd280_hits import ND280HitsReader
from .readers.nd280_labl import ND280LablReader


@DATASETS.register_module()
class ND280Dataset(ShardEventDataset):
    """T2K ND280 multimodal dataset with nested per-modality output.

    Parameters
    ----------
    data_root : str
        Root directory containing per-modality subdirectories.
    split : str
        Split name for file discovery.
    modalities : tuple[str]
        Any subset of ``{'step', 'sensor', 'hits', 'labl'}``.
        ``('labl',)`` and ``('sensor', 'labl')`` are invalid.
    dataset_name : str
        File prefix (``'nd280'`` matches ``nd280_step_0000.h5``).
    subdets : iterable[int] or None
        Keep only these sub-detector codes (ND::GeomId::Def: 1 P0D, 2 TPC,
        3 FGD, 4 DsECal, 5 BrlECal, 6 P0DECal, 7 SMRD, 9 SuperFGD, 10 HA-TPC,
        11 ToF) in step / sensor / hits. ``None`` keeps all.
    include_noise : bool
        Keep the noise rows of ``hits`` (``contribution_type == 2``).
    min_segments : int
        Drop events with fewer step segments (step only).
    include_physics : bool
        Step also emits ``start``, ``end``, ``energy_vis``, ``length``.
    include_waveforms : bool
        Sensor also emits the TPC/HA-TPC waveform samples.
    labels, label_config : label decoration (see :mod:`pimm_data._label_decorate`).
    transform, loop, max_len, ignore_index, strict_lengths : standard parameters.
    """

    def __init__(
        self,
        data_root,
        split='',
        modalities=('sensor',),
        dataset_name='nd280',
        subdets=None,
        include_noise=True,
        min_segments=0,
        include_physics=True,
        include_waveforms=False,
        labels=None,
        label_config=None,
        transform=None,
        loop=1,
        max_len=-1,
        ignore_index=-1,
        strict_lengths=False,
    ):
        self._modalities = tuple(modalities)
        self._validate_modalities(self._modalities)
        if min_segments > 0 and 'step' not in self._modalities:
            raise ValueError(
                f"min_segments={min_segments} filters on step segment counts "
                f"but modalities={self._modalities} does not include 'step'.")
        if isinstance(labels, dict):
            label_config = labels.get('config', label_config)
        self._want_labels = (labels is not None) or ('labl' in self._modalities)
        if self._want_labels and not ({'step', 'hits'} & set(self._modalities)):
            raise ValueError(
                "labels were requested (labels=/'labl') but modalities has no "
                "decoratable point cloud — need 'step' or 'hits'; "
                f"modalities={self._modalities}.")

        self._dataset_name = dataset_name
        self._subdets = None if subdets is None else np.array(sorted(set(subdets)), dtype=np.uint8)
        self._label_config = label_config
        self._max_len = max_len
        self._strict_lengths = strict_lengths
        self._source_data_root = data_root
        self._source_split = split
        self._ignore = ignore_index

        self.step_reader = None
        self.sensor_reader = None
        self.hits_reader = None
        self.labl_reader = None
        if 'step' in self._modalities:
            self.step_reader = ND280StepReader(
                data_root=self._modality_root('step'), split=split,
                dataset_name=dataset_name, min_segments=min_segments,
                include_physics=include_physics)
        if 'sensor' in self._modalities:
            self.sensor_reader = ND280SensorReader(
                data_root=self._modality_root('sensor'), split=split,
                dataset_name=dataset_name, include_waveforms=include_waveforms)
        if 'hits' in self._modalities:
            self.hits_reader = ND280HitsReader(
                data_root=self._modality_root('hits'), split=split,
                dataset_name=dataset_name, include_noise=include_noise)
        if self._want_labels:
            self.labl_reader = ND280LablReader(
                data_root=self._modality_root('labl'), split=split,
                dataset_name=dataset_name)

        self._canonical_reader = (self.step_reader or self.hits_reader
                                  or self.sensor_reader or self.labl_reader)
        self._build_joint_index(
            source_label=f"ND280Dataset({data_root!r})",
            filter_label=(f"min_segments={min_segments}" if min_segments > 0 else ''))
        super().__init__(split=split, data_root=data_root, transform=transform,
                         ignore_index=ignore_index, loop=loop)

    # ------------------------------------------------------------------

    def get_data(self, idx):
        real_idx = idx % len(self.data_list)
        data = {
            'name': self.get_data_name(real_idx),
            'split': self.split if isinstance(self.split, str) else 'custom',
        }
        labl = None
        if self.labl_reader is not None:
            labl = self._build_labl(self.labl_reader.read_event(real_idx))
            data['labl'] = labl
        if self.sensor_reader is not None:
            data['sensor'] = self._build_sensor(self.sensor_reader.read_event(real_idx))
        if self.hits_reader is not None:
            data['hits'] = self._build_hits(self.hits_reader.read_event(real_idx), labl)
        if self.step_reader is not None:
            data['step'] = self._build_step(self.step_reader.read_event(real_idx), labl)
        return data

    def _keep(self, subdet):
        if self._subdets is None:
            return None
        return np.isin(subdet, self._subdets)

    @staticmethod
    def _sensor_columns(table, sensor_idx):
        return {
            'coord': table['positions'][sensor_idx].astype(np.float32),
            'subdet': table['subdet'][sensor_idx].astype(np.uint8),
            'axis': table['axis'][sensor_idx].astype(np.uint8),
            'half_size': table['half_size'][sensor_idx].astype(np.float32),
        }

    def _build_sensor(self, raw):
        sidx = raw['sensor_idx']
        sub = self._sensor_columns(raw['sensor_table'], sidx)
        sub.update({'energy': raw['Q'][:, None], 'time': raw['T'][:, None], 'sensor_idx': sidx})
        keep = self._keep(sub['subdet'])
        if 'wave_offsets' in raw:
            sub['wave_offsets'], sub['wave_q'], sub['wave_t'] = raw['wave_offsets'], raw['wave_q'], raw['wave_t']
        if keep is not None:
            if 'wave_offsets' in sub:  # re-pack the kept rows' samples
                off = sub['wave_offsets']
                lengths = np.diff(off)[keep]
                idx = np.concatenate([np.arange(off[i], off[i + 1]) for i in np.nonzero(keep)[0]]) \
                    if keep.any() else np.zeros(0, dtype=np.int64)
                sub['wave_offsets'] = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int32)
                sub['wave_q'], sub['wave_t'] = sub['wave_q'][idx], sub['wave_t'][idx]
            sub = {k: (v[keep] if not k.startswith('wave_') else v) for k, v in sub.items()}
        return sub

    def _build_hits(self, raw, labl):
        pidx = raw['particle_idx']
        sub = self._sensor_columns(raw['sensor_table'], raw['sensor_idx'])
        sub.update({
            'energy': raw['Q'][:, None], 'time': raw['T_true'][:, None],
            'sensor_idx': raw['sensor_idx'], 'digit_idx': raw['digit_idx'],
            'contribution_type': raw['contribution_type'],
            'particle_idx': pidx, 'instance': pidx.astype(np.int32),
        })
        if labl is not None:
            pdg = labl['particle'].get('pdg')
            if pdg is not None:
                sub['segment'] = gather_with_fill(pidx, pdg, fill=self._ignore)
            if self._label_config is not None:
                decorate_labels(sub, labl, lambda n: (pidx if n == 'particle_idx' else None),
                                self._label_config)
        keep = self._keep(sub['subdet'])
        return sub if keep is None else {k: v[keep] for k, v in sub.items()}

    def _build_step(self, raw, labl):
        sub = dict(raw)
        if labl is not None:
            track_pidx = labl['track'].get('particle_idx')
            if track_pidx is not None:
                pidx = gather_with_fill(sub['track_idx'], track_pidx)
                sub['particle_idx'] = pidx
                sub['instance'] = pidx
                pdg = labl['particle'].get('pdg')
                if pdg is not None:
                    sub['segment'] = gather_with_fill(pidx, pdg, fill=self._ignore)
            if self._label_config is not None:
                pidx = sub.get('particle_idx')
                decorate_labels(sub, labl, lambda n: (pidx if n == 'particle_idx' else None),
                                self._label_config)
        keep = self._keep(sub['subdet'])
        return sub if keep is None else {k: v[keep] for k, v in sub.items()}

    def _build_labl(self, flat):
        out = {'event': {}, 'interaction': {}, 'particle': {}, 'track': {}}
        for k, v in flat.items():
            for scope in out:
                prefix = f'labl_{scope}_'
                if k.startswith(prefix):
                    out[scope][k[len(prefix):]] = v
                    break
        return out

    # ------------------------------------------------------------------

    @property
    def detector_geometry(self):
        """Geometry from the data: ``baseline``, ``geometry_hash``,
        ``volumes`` (drawing boxes) and ``sensor_table``; ``None`` values when
        no loaded reader provides them."""
        geom = dict(baseline=None, geometry_hash=None, volumes=None, sensor_table=None)
        if self.step_reader is not None:
            geom.update(self.step_reader.detector_geometry)
        for reader in (self.sensor_reader, self.hits_reader):
            if reader is not None and reader.sensor_table is not None:
                geom['sensor_table'] = reader.sensor_table
                break
        return geom
