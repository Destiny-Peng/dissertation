"""Localization analysis workflow composition."""

from analysis_localization_challenge import AnalysisLocalizationChallengeMixin
from analysis_localization_results import AnalysisLocalizationResultsMixin


class AnalysisLocalizationMixin(
    AnalysisLocalizationResultsMixin,
    AnalysisLocalizationChallengeMixin,
):
    """Compose localization result and challenge-set workflows."""

    pass
