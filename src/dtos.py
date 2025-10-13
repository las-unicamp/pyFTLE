from typing import List, TypedDict


class FTLETask(TypedDict):
    snapshots: List[str]
    coordinates: List[str]
    particles: str  # Assume a single particle file
