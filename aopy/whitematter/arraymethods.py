import numpy as np

class MetadataArray(np.ndarray):
    """Numpy array plus metadata."""

    def __new__(cls, input_array, metadata=None):
        obj = np.asarray(input_array).view(cls)
        obj.metadata = metadata
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.metadata = getattr(obj, 'metadata', None)