from app.constants.series_types.sdk.metric_types import SAMSUNG_METRIC_TYPE_TO_SERIES_TYPE
from app.constants.series_types.sdk.workout_statistics import WORKOUT_STATISTIC_TYPE_TO_SERIES_TYPE
from app.schemas.enums import SeriesType
from app.services.providers.apple.coverage import HEALTH_SCORES, MEAL_FIELDS, SLEEP_FIELDS, WORKOUT_FIELDS

# Samsung Health emits Android/HC metric types (RMSSD, not SDNN), minus the dietary
# types the Samsung SDK never sends (no caffeine, chloride, or extended vitamins/minerals).
TIMESERIES: frozenset[SeriesType] = frozenset(
    {
        *SAMSUNG_METRIC_TYPE_TO_SERIES_TYPE.values(),
        *WORKOUT_STATISTIC_TYPE_TO_SERIES_TYPE.values(),
    }
)

__all__ = ["HEALTH_SCORES", "MEAL_FIELDS", "SLEEP_FIELDS", "TIMESERIES", "WORKOUT_FIELDS"]
