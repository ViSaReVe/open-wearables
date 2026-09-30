"""
Tests for DataSourceRepository.

Regression coverage for the ``data_source.source`` column width: Apple
HealthKit tags on-device data with a source bundle identifier of the form
``com.apple.health.<UUID>`` (53 chars). This overflowed the previous
``VARCHAR(50)`` and aborted the entire SDK import batch with
``StringDataRightTruncation``, so no Apple sleep/records were ever saved.
The column is now ``VARCHAR(100)``.
"""

from sqlalchemy.orm import Session

from app.models import DataSource
from app.repositories.data_source_repository import DataSourceRepository
from app.schemas.enums import DeviceType, ProviderName
from tests.factories import UserFactory

# Realistic Apple HealthKit source bundle id: "com.apple.health." + a UUID.
APPLE_HEALTH_SOURCE = "com.apple.health.ED447642-08FD-4E45-AF20-633C02C83170"


class TestDataSourceRepository:
    """Test suite for DataSourceRepository."""

    def test_apple_health_source_bundle_id_persists(self, db: Session) -> None:
        """A 53-char Apple source bundle id imports without truncation.

        Exercises ``ensure_data_source`` (the SDK import path that failed) and
        asserts the full identifier round-trips — this would raise
        ``StringDataRightTruncation`` against the old ``VARCHAR(50)`` column.
        """
        # Arrange: a source longer than the old 50-char limit.
        assert len(APPLE_HEALTH_SOURCE) > 50
        user = UserFactory()
        repo = DataSourceRepository(DataSource)

        # Act
        created = repo.ensure_data_source(
            db,
            user_id=user.id,
            provider=ProviderName.APPLE,
            device_model="Watch7,5",
            source=APPLE_HEALTH_SOURCE,
        )
        db.commit()
        db.expire_all()

        # Assert: stored intact, retrievable by its full identity.
        assert created.source == APPLE_HEALTH_SOURCE
        stored = repo.get_by_identity(
            db,
            user_id=user.id,
            provider=ProviderName.APPLE,
            device_model="Watch7,5",
            source=APPLE_HEALTH_SOURCE,
        )
        assert stored is not None
        assert stored.source == APPLE_HEALTH_SOURCE
        assert len(stored.source) == len(APPLE_HEALTH_SOURCE)

    def test_sdk_device_type_upgrades_from_other_but_never_flips(self, db: Session) -> None:
        user = UserFactory()
        repo = DataSourceRepository(DataSource)
        identity = {
            "user_id": user.id,
            "provider": ProviderName.SAMSUNG,
            "device_model": "unlisted-model",
            "source": "tab",
        }
        created = repo.ensure_data_source(db, **identity)
        assert created.device_type == DeviceType.OTHER

        repo.ensure_data_source(db, **identity, reported_type=DeviceType.WATCH)
        assert repo.get_by_identity(db, **identity).device_type == DeviceType.WATCH

        repo.batch_ensure_data_sources(
            db,
            ProviderName.SAMSUNG,
            None,
            {(user.id, "unlisted-model", "tab", None, None)},
            {(user.id, "unlisted-model", "tab", None, None): DeviceType.BAND},
        )
        assert repo.get_by_identity(db, **identity).device_type == DeviceType.WATCH

    def test_cloud_device_type_is_corrected_on_sync(self, db: Session) -> None:
        user = UserFactory()
        repo = DataSourceRepository(DataSource)
        identity = {
            "user_id": user.id,
            "provider": ProviderName.GARMIN,
            "device_model": "Garmin Index BPM",
            "source": "garmin",
        }
        stored = repo.ensure_data_source(db, **identity)
        object.__setattr__(stored, "device_type", DeviceType.SCALE.value)
        db.flush()

        repo.ensure_data_source(db, **identity)
        assert repo.get_by_identity(db, **identity).device_type == DeviceType.BP_MONITOR

    def test_batch_upgrades_unset_device_type(self, db: Session) -> None:
        user = UserFactory()
        repo = DataSourceRepository(DataSource)
        identity = (user.id, None, "com.fitbit.FitbitMobile", None, None)
        repo.batch_ensure_data_sources(db, ProviderName.HEALTH_CONNECT, None, {identity})
        ds = repo.get_by_identity(db, user.id, ProviderName.HEALTH_CONNECT, None, "com.fitbit.FitbitMobile")
        assert ds.device_type is None

        repo.batch_ensure_data_sources(db, ProviderName.HEALTH_CONNECT, None, {identity}, {identity: DeviceType.BAND})
        db.expire_all()
        ds = repo.get_by_identity(db, user.id, ProviderName.HEALTH_CONNECT, None, "com.fitbit.FitbitMobile")
        assert ds.device_type == DeviceType.BAND

    def test_batch_resolves_stored_empty_model_and_stores_software_version(self, db: Session) -> None:
        user = UserFactory()
        repo = DataSourceRepository(DataSource)
        stored = repo.ensure_data_source(
            db, user_id=user.id, provider=ProviderName.STRAVA, device_model="", source="strava"
        )
        identity = (user.id, None, "strava", None, None)
        new_identity = (user.id, "SM-L315F", "Galaxy Watch7", None, None)

        result = repo.batch_ensure_data_sources(db, ProviderName.STRAVA, None, {identity})
        repo.batch_ensure_data_sources(db, ProviderName.SAMSUNG, None, {new_identity}, None, {new_identity: "5.0.1"})

        assert result[identity] == stored.id
        created = repo.get_by_identity(db, user.id, ProviderName.SAMSUNG, "SM-L315F", "Galaxy Watch7")
        assert created.software_version == "5.0.1"

    def test_device_id_links_records_and_fills_model(self, db: Session) -> None:
        user = UserFactory()
        repo = DataSourceRepository(DataSource)
        sleep = repo.ensure_data_source(
            db, user_id=user.id, provider=ProviderName.POLAR, source="polar", device_id="15995F33"
        )
        workout = repo.ensure_data_source(
            db,
            user_id=user.id,
            provider=ProviderName.POLAR,
            device_model="Polar Vantage M3",
            source="polar",
            device_id="15995F33",
        )

        assert workout.id == sleep.id
        assert workout.device_model == "Polar Vantage M3"

    def test_stable_ids_are_stamped_onto_legacy_row(self, db: Session) -> None:
        user = UserFactory()
        repo = DataSourceRepository(DataSource)
        legacy = repo.ensure_data_source(
            db, user_id=user.id, provider=ProviderName.APPLE, device_model="Watch6,2", source="Apple Watch"
        )

        keyed = repo.ensure_data_source(
            db,
            user_id=user.id,
            provider=ProviderName.APPLE,
            device_model="Watch6,2",
            source="Apple Watch",
            app_id="com.apple.health.X",
        )
        renamed = repo.ensure_data_source(
            db,
            user_id=user.id,
            provider=ProviderName.APPLE,
            device_model="Watch6,2",
            source="Kuba's Watch",
            app_id="com.apple.health.X",
        )

        assert keyed.id == legacy.id == renamed.id
        assert keyed.app_id == "com.apple.health.X"

    def test_ids_ignored_for_providers_without_stable_ids(self, db: Session) -> None:
        user = UserFactory()
        repo = DataSourceRepository(DataSource)
        ds = repo.ensure_data_source(
            db,
            user_id=user.id,
            provider=ProviderName.GARMIN,
            device_model="Garmin fenix 8",
            source="garmin",
            device_id="x",
        )

        assert ds.device_id is None
