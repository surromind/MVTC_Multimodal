"""Core modules for multimodal concatenation pipeline."""

from .concat import ConcatenationFusion
from .config_loader import ProjectConfigDict, load_project_config
from .data import (
    MultimodalDataset,
    TactileNormalizer,
    SampleKeyDict,
    MultimodalSampleDict,
)
from .embedders import VTCEmbedder, CLIPEmbedder, BERTEmbedder, EmbedOutputsDict
from .scaler import FeatureScaler
from .tactile import TactileEncoder, TimeSeriesAutoencoder, load_tactile_encoder
from .visualization import (
    save_all,
    save_tsne,
    save_umap,
    VisualizationResultDict,
)
from app.utils.path import DatasetPaths, ModelPathsDict

__all__ = [
    "ConcatenationFusion",
    "MultimodalDataset",
    "TactileNormalizer",
    "SampleKeyDict",
    "MultimodalSampleDict",
    "VTCEmbedder",
    "CLIPEmbedder",
    "BERTEmbedder",
    "EmbedOutputsDict",
    "FeatureScaler",
    "TactileEncoder",
    "TimeSeriesAutoencoder",
    "save_tsne",
    "save_umap",
    "save_all",
    "VisualizationResultDict",
    "ProjectConfigDict",
    "load_project_config",
    "DatasetPaths",
    "ModelPathsDict",
    "load_tactile_encoder",
]
