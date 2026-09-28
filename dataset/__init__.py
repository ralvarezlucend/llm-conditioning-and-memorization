"""
Datasets package for memorization experiments.

This package provides various dataset implementations for studying
memorization in language models.
"""

from dataset.base import (
    BaseHybridDataset,
    BaseRepeatedNovelDataset,
    generate_lengths_half_gaussian,
    generate_random_sequences_half_gaussian,
    generate_random_sequences_fixed_uniform,
)
from dataset.dataloader import BatchBasedDataLoader
from dataset.wikitext_synthetic import WikitextSyntheticDataset
from dataset.novel_repeated_random import NovelRepeatedRandomDataset
from dataset.repeated_random_fixed import RepeatedRandomFixedLengthDataset
from dataset.variable_vocab import VariableVocabDataset
from dataset.three_region_variable_vocab import ThreeRegionVariableVocabDataset
from dataset.wikitext import WikitextDataset
from dataset.wikitext_original_shuffled import WikitextOriginalShuffledDataset
from dataset.npz_dataset import NpzDataset
from dataset.wikitext_simple import WikitextSimpleDataset

__all__ = [
    # Base classes and utilities
    "BaseHybridDataset",
    "BaseRepeatedNovelDataset",
    "generate_lengths_half_gaussian",
    "generate_random_sequences_half_gaussian",
    "generate_random_sequences_fixed_uniform",
    # Dataset implementations
    "NovelRepeatedRandomDataset",
    "RepeatedRandomFixedLengthDataset",
    "VariableVocabDataset",
    "ThreeRegionVariableVocabDataset",
    "WikitextDataset",
    "WikitextSyntheticDataset",
    "WikitextOriginalShuffledDataset",
    "NpzDataset",
    "WikitextSimpleDataset",
    # Data loader
    "BatchBasedDataLoader",
]
