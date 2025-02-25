import enum

class PreparationType(enum.Enum):
    """Magnetization preparation type."""
    NONE = 0
    INVERSION = 1
    T2_PREPARATION = 2
