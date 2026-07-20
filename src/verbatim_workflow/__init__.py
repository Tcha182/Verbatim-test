"""Mistral durable workflow for multi-label verbatim classification.

Public surface kept import-light on purpose: importing this package must NOT
import the Mistral SDK, so the pure logic (config, schemas, prompts, clustering)
stays testable without network/credentials. The SDK is only imported inside the
modules that actually talk to Mistral (``mistral_client``, ``activities``,
``workflow``, ``worker``, ``run_local``).
"""

from .config import PipelineConfig, Category, Settings, DiscoveryConfig, load_config
from .schemas import build_classification_model, Confidence, ProposedCategory

__all__ = [
    "PipelineConfig",
    "Category",
    "Settings",
    "DiscoveryConfig",
    "load_config",
    "build_classification_model",
    "Confidence",
    "ProposedCategory",
]
