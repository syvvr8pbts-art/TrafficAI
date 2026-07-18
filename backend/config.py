"""
backend/config.py

Backend configuration -- the single source of truth for *server/serving-layer*
settings (HTTP host/port, debug flag). It deliberately does NOT redefine any
detection parameters: those already live in the project-root ``config.py``
(``AppConfig`` / ``DetectorConfig``), and this module simply re-exposes them so
there is exactly one place each value is defined.

Anything that used to be hardcoded in the serving layer (e.g. host/port) belongs
here, config-driven, so no other backend module hardcodes it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Root project configuration (unchanged). Imported absolutely so it resolves to
# the top-level config.py, not this package's config module.
from config import AppConfig, DetectorConfig, load_config


@dataclass
class ServerConfig:
    """HTTP server settings for the API layer."""
    host: str = "127.0.0.1"
    port: int = 8000
    debug: bool = False


@dataclass
class BackendConfig:
    """Aggregate backend configuration.

    ``app`` is the existing root application config, so all detector settings
    (weights path, device, thresholds, class filter) flow through unchanged.
    """
    server: ServerConfig = field(default_factory=ServerConfig)
    app: AppConfig = field(default_factory=load_config)

    @property
    def detector(self) -> DetectorConfig:
        """Convenience accessor for the root detector configuration."""
        return self.app.detector


def get_config() -> BackendConfig:
    """Return a fresh BackendConfig (defaults + root AppConfig)."""
    return BackendConfig()
