from dataclasses import dataclass
from abc import ABC
from typing import List, Iterable

import numpy as np


class DrawData(ABC):
    pass


@dataclass
class CurveData(DrawData):
    points: np.ndarray


@dataclass
class VectorData(DrawData):
    point: np.ndarray
    vector: np.ndarray

@dataclass
class FrameData(DrawData):
    point: np.ndarray
    vecs: Iterable[np.ndarray]

@dataclass
class DrawDataList(DrawData):
    data_list: List[DrawData]

    @classmethod
    def new(cls):
        return cls([])

    def append(self, data: DrawData):
        self.data_list.append(data)

@dataclass
class ConeData(DrawData):
    apex: np.ndarray
    vecs: Iterable[np.ndarray]