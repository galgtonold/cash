"""Abstract data source interface for cache-aware data loading."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod

__all__ = ["DataSource", "FileDataSource"]

class DataSource(ABC):
    """Abstract base class for data sources."""

    @abstractmethod
    def get_id(self) -> str:
        """Unique identifier for the data source."""

    @abstractmethod
    def has_changed(self) -> bool:
        """Check if the data source has changed since last check."""

    @abstractmethod
    def update_state(self) -> None:
        """Update the internal state to current (mark as seen)."""

class FileDataSource(DataSource):
    """Tracks a file for changes using modification time."""

    def __init__(self, filepath: str):
        self.filepath = os.path.abspath(filepath)
        self._last_mtime = self._get_mtime()

    def _get_mtime(self) -> float:
        try:
            return os.path.getmtime(self.filepath)
        except OSError:
            return 0.0

    def get_id(self) -> str:
        return f"file:{self.filepath}"

    def has_changed(self) -> bool:
        current_mtime = self._get_mtime()
        return current_mtime != self._last_mtime

    def update_state(self) -> None:
        self._last_mtime = self._get_mtime()
