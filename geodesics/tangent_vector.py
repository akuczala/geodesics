from dataclasses import dataclass
from enum import Enum

import numpy as np

from geodesics.constants import EPSILON_SQ


@dataclass
class TangentVector:
    x: np.ndarray
    u: np.ndarray


class TangentVectorType(Enum):
    TIMELIKE = 1
    SPACELIKE = -1
    NULL = 0

    @classmethod
    def len_to_type(cls, sqlength: float) -> "TangentVectorType":
        if sqlength > EPSILON_SQ:
            return cls.TIMELIKE
        elif sqlength < -EPSILON_SQ:
            return cls.SPACELIKE
        else:
            return cls.NULL
