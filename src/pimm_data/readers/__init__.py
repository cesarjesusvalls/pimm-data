"""Readers for detector HDF5 files.

Each reader is self-contained: given a data_root and split, it discovers
shard files, builds an event index, and exposes ``read_event(idx)``
returning a flat ``dict[str, np.ndarray]``.
"""

from .jaxtpc_step import JAXTPCStepReader
from .jaxtpc_sensor import JAXTPCSensorReader
from .jaxtpc_labl import JAXTPCLablReader
from .jaxtpc_hits import JAXTPCHitsReader
from .lucid_step import LUCiDStepReader
from .lucid_sensor import LUCiDSensorReader
from .lucid_hits import LUCiDHitsReader
from .lucid_labl import LUCiDLablReader
from .nd280_step import ND280StepReader
from .nd280_sensor import ND280SensorReader
from .nd280_hits import ND280HitsReader
from .nd280_labl import ND280LablReader
from .optical_sensor import OpticalSensorReader, OpticalEastWestReader

__all__ = [
    "JAXTPCStepReader",
    "JAXTPCSensorReader",
    "JAXTPCLablReader",
    "JAXTPCHitsReader",
    "LUCiDStepReader",
    "LUCiDSensorReader",
    "LUCiDHitsReader",
    "LUCiDLablReader",
    "ND280StepReader",
    "ND280SensorReader",
    "ND280HitsReader",
    "ND280LablReader",
    "OpticalSensorReader",
    "OpticalEastWestReader",
]
