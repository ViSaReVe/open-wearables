"""Stable data source identifiers per provider.

A field is listed only if that provider's ingest path always sends it, so the identity of a
device never flips between keyed and unkeyed records.
"""

from app.schemas.enums import ProviderName

STABLE_ID_FIELDS: dict[ProviderName, frozenset[str]] = {
    ProviderName.APPLE: frozenset({"source_app_id"}),
    ProviderName.HEALTH_CONNECT: frozenset({"source_app_id"}),
    ProviderName.SAMSUNG: frozenset({"device_id", "source_app_id"}),
    ProviderName.POLAR: frozenset({"device_id"}),
}


def stable_ids(
    provider: ProviderName, device_id: str | None, source_app_id: str | None
) -> tuple[str | None, str | None]:
    """(device_id, source_app_id) kept only where the provider's identity rules allow them."""
    fields = STABLE_ID_FIELDS.get(provider, frozenset())
    device_id = (device_id or "").strip() or None
    source_app_id = (source_app_id or "").strip() or None
    return (device_id if "device_id" in fields else None, source_app_id if "source_app_id" in fields else None)
