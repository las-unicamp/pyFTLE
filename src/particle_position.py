import numpy as np

from src.dtos import NeighboringParticles


class PositionDict:
    """Operator overloading for NeighboringParticles dataclass"""

    def __init__(self, data: NeighboringParticles):
        self.data: NeighboringParticles = data

    def __add__(self, other):
        """Element-wise addition with another PositionDict or a scalar/array."""
        if isinstance(other, PositionDict):
            return PositionDict(
                NeighboringParticles(
                    left=self.data.left + other.data.left,
                    right=self.data.right + other.data.right,
                    top=self.data.top + other.data.top,
                    bottom=self.data.bottom + other.data.bottom,
                )
            )
        elif isinstance(other, (int, float, np.ndarray)):
            return PositionDict(
                NeighboringParticles(
                    left=self.data.left + other,
                    right=self.data.right + other,
                    top=self.data.top + other,
                    bottom=self.data.bottom + other,
                )
            )
        raise TypeError(f"Unsupported type {type(other)} for addition.")

    def __mul__(self, other):
        """Element-wise multiplication with a scalar/array."""
        if isinstance(other, (int, float, np.ndarray)):
            return PositionDict(
                NeighboringParticles(
                    left=self.data.left * other,
                    right=self.data.right * other,
                    top=self.data.top * other,
                    bottom=self.data.bottom * other,
                )
            )
        raise TypeError(f"Unsupported type {type(other)} for multiplication.")

    def to_array(self) -> np.ndarray:
        """Flatten and concatenate dataclass values into a single NumPy array."""
        return np.concatenate(
            [self.data.left, self.data.right, self.data.top, self.data.bottom]
        )

    @classmethod
    def from_array(cls, array: np.ndarray, template: NeighboringParticles):
        """Reconstruct PositionDict from a NumPy array using a template."""
        size = template.top.shape[0]
        left = array[:size].reshape(template.left.shape)
        right = array[size : size * 2].reshape(template.right.shape)
        top = array[size * 2 : size * 3].reshape(template.top.shape)
        bottom = array[size * 3 :].reshape(template.bottom.shape)

        reconstructed = NeighboringParticles(
            left=left, right=right, top=top, bottom=bottom
        )
        return cls(reconstructed)

    def __repr__(self):
        """
        String representation of the PositionDict object.
        Uses numpy array formatting to handle large arrays.
        """
        formatted_items = [
            f"    '{key}': "
            + np.array2string(getattr(self.data, key), threshold=5).replace(
                "\n", "\n           "
            )
            for key in ["left", "right", "top", "bottom"]
        ]
        return "PositionDict({\n" + ",\n".join(formatted_items) + "\n})"

    def __len__(self):
        """Returns the number of particles, based on the 'top' array."""
        return self.data.top.shape[0]
