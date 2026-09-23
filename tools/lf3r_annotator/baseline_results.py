"""Baseline result-side service composition."""

from baseline_catalog import BaselineCatalogMixin
from baseline_readers import BaselineReadersMixin


class BaselineResultsMixin(
    BaselineReadersMixin,
    BaselineCatalogMixin,
):
    """Compose baseline run readers with catalog/coverage behavior."""

    pass
