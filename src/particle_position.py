import numpy as np

from src.dtos import NeighboringParticles


class PositionDict:
    """Operator overloading for NeighboringParticles TypedDict"""

    def __init__(self, position: NeighboringParticles):
        self.position: NeighboringParticles = position

    def __add__(self, other):
        """Element-wise addition with another PositionDict or a scalar/array."""
        if isinstance(other, PositionDict):
            return PositionDict(
                {key: self.position[key] + other.position[key] for key in self.position}
            )  # type: ignore
        elif isinstance(other, (int, float, np.ndarray)):
            return PositionDict(
                {key: value + other for key, value in self.position.items()}
            )  # type: ignore
        raise TypeError(f"Unsupported type {type(other)} for addition.")

    def __mul__(self, other):
        """Element-wise multiplication with a scalar/array."""
        if isinstance(other, (int, float, np.ndarray)):
            return PositionDict(
                {key: value * other for key, value in self.position.items()}
            )  # type: ignore
        raise TypeError(f"Unsupported type {type(other)} for multiplication.")

    def to_array(self) -> np.ndarray:
        """Flatten and concatenate dictionary values into a single NumPy array."""
        return np.concatenate([self.position[key] for key in self.position])

    @classmethod
    def from_array(cls, array: np.ndarray, template: NeighboringParticles):
        """Reconstruct PositionDict from a NumPy array using a template."""
        reconstructed: NeighboringParticles = {}
        start = 0
        for key, value in template.items():
            size = value.shape[0]  # Maintain shape (Nx2)
            reconstructed[key] = array[start : start + size].reshape(value.shape)
            start += size
        return cls(reconstructed)

    def __repr__(self):
        return f"PositionDict({self.position})"
