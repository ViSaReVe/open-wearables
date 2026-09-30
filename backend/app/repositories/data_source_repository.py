from typing import cast
from uuid import UUID, uuid4

from sqlalchemy import CursorResult, and_, asc, delete, func
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.sql.elements import ColumnElement

from app.constants.devices_map import infer_device_type
from app.constants.devices_map.data_source_identity import stable_ids
from app.constants.sdk_providers import sdk_providers
from app.database import DbSession
from app.models import DataSource, HealthScore, ProviderPriority
from app.repositories.provider_priority_repository import ProviderPriorityRepository
from app.repositories.repositories import CrudRepository
from app.schemas.enums import DeviceType, ProviderName
from app.schemas.model_crud.data_priority import DataSourceCreate, DataSourceUpdate

# (user_id, device_model, source, device_id, app_id)
DataSourceIdentity = tuple[UUID, str | None, str | None, str | None, str | None]


class DataSourceRepository(
    CrudRepository[DataSource, DataSourceCreate, DataSourceUpdate],
):
    def __init__(self, model: type[DataSource] = DataSource):
        super().__init__(model)

    def _legacy_filter(
        self,
        user_id: UUID,
        provider: ProviderName,
        device_model: str | None,
        source: str | None,
    ) -> ColumnElement[bool]:
        return and_(
            self.model.user_id == user_id,
            self.model.provider == provider,
            self.model.device_id.is_(None),
            self.model.app_id.is_(None),
            func.coalesce(self.model.device_model, "") == (device_model or ""),
            func.coalesce(self.model.source, "") == (source or ""),
        )

    def _find(self, db_session: DbSession, provider: ProviderName, identity: DataSourceIdentity) -> DataSource | None:
        """Look up by the most stable identifier present: device id, then app id + model, then model + source."""
        user_id, device_model, source, device_id, app_id = identity
        query = db_session.query(self.model).filter(self.model.user_id == user_id, self.model.provider == provider)
        if device_id:
            return query.filter(self.model.device_id == device_id).one_or_none()
        if app_id:
            return query.filter(
                self.model.device_id.is_(None),
                self.model.app_id == app_id,
                func.coalesce(self.model.device_model, "") == (device_model or ""),
            ).one_or_none()
        return (
            db_session.query(self.model)
            .filter(self._legacy_filter(user_id, provider, device_model, source))
            .one_or_none()
        )

    def _stampable(
        self, db_session: DbSession, provider: ProviderName, identity: DataSourceIdentity
    ) -> DataSource | None:
        """Row written before its stable ids were known (app-id row for a new device id, else a legacy row)."""
        user_id, device_model, source, device_id, app_id = identity
        if (
            device_id
            and app_id
            and (row := self._find(db_session, provider, (user_id, device_model, source, None, app_id)))
        ):
            return row
        if device_id or app_id:
            return self._find(db_session, provider, (user_id, device_model, source, None, None))
        return None

    def get_by_identity(
        self,
        db_session: DbSession,
        user_id: UUID,
        provider: ProviderName,
        device_model: str | None = None,
        source: str | None = None,
        device_id: str | None = None,
        app_id: str | None = None,
    ) -> DataSource | None:
        return self._find(db_session, provider, (user_id, device_model, source, device_id, app_id))

    def ensure_data_source(
        self,
        db_session: DbSession,
        user_id: UUID,
        provider: ProviderName,
        user_connection_id: UUID | None = None,
        device_model: str | None = None,
        software_version: str | None = None,
        source: str | None = None,
        original_source_name: str | None = None,
        reported_type: DeviceType | None = None,
        device_id: str | None = None,
        app_id: str | None = None,
    ) -> DataSource:
        return self._resolve(
            db_session,
            provider,
            (user_id, device_model, source, *stable_ids(provider, device_id, app_id)),
            user_connection_id=user_connection_id,
            software_version=software_version,
            original_source_name=original_source_name,
            reported_type=reported_type,
        )

    def _resolve(
        self,
        db_session: DbSession,
        provider: ProviderName,
        identity: DataSourceIdentity,
        user_connection_id: UUID | None = None,
        software_version: str | None = None,
        original_source_name: str | None = None,
        reported_type: DeviceType | None = None,
    ) -> DataSource:
        user_id, device_model, source, device_id, app_id = identity
        existing = self._find(db_session, provider, identity) or self._stampable(db_session, provider, identity)
        if existing is None:
            ProviderPriorityRepository(ProviderPriority).ensure_provider_exists(db_session, provider)
            device_type = infer_device_type(provider, device_model, original_source_name or source, reported_type)
            stmt = (
                insert(self.model)
                .values(
                    id=uuid4(),
                    user_id=user_id,
                    provider=provider,
                    user_connection_id=user_connection_id,
                    device_model=device_model,
                    software_version=software_version,
                    source=source,
                    device_type=device_type.value if device_type != DeviceType.UNKNOWN else None,
                    original_source_name=original_source_name,
                    device_id=device_id,
                    app_id=app_id,
                )
                .on_conflict_do_nothing()
            )
            db_session.execute(stmt)
            db_session.flush()
            created = self._find(db_session, provider, identity)
            assert created is not None
            return created

        updates: dict[str, object] = {}
        if device_id and existing.device_id is None:
            updates["device_id"] = device_id
        if app_id and existing.app_id is None:
            updates["app_id"] = app_id
        # Device-keyed rows can learn their model later (e.g. Polar sleep before the exercise)
        if device_id and device_model and existing.device_model is None:
            updates["device_model"] = device_model
        if user_connection_id and existing.user_connection_id is None:
            updates["user_connection_id"] = user_connection_id
        if software_version and existing.software_version is None:
            updates["software_version"] = software_version
        if original_source_name and existing.original_source_name is None:
            updates["original_source_name"] = original_source_name
        device_type = self.next_device_type(
            provider,
            existing.device_type,
            infer_device_type(
                provider,
                device_model or existing.device_model,
                original_source_name or existing.original_source_name or existing.source,
                reported_type,
            ),
        )
        if device_type != existing.device_type:
            updates["device_type"] = device_type
        for field, value in updates.items():
            object.__setattr__(existing, field, value)
        if updates:
            db_session.flush()
        return existing

    @staticmethod
    def next_device_type(provider: ProviderName, current: str | None, resolved: DeviceType) -> str | None:
        """Cloud rows take the inferred type; SDK rows only upgrade from unset/"other"."""
        if provider.value not in sdk_providers():
            return resolved.value if resolved != DeviceType.UNKNOWN else None
        if current in (None, DeviceType.OTHER) and resolved not in (DeviceType.UNKNOWN, current):
            return resolved.value
        return current

    def batch_ensure_data_sources(
        self,
        db_session: DbSession,
        provider: ProviderName,
        user_connection_id: UUID | None,
        identities: set[DataSourceIdentity],
        reported_types: dict[DataSourceIdentity, DeviceType] | None = None,
        software_versions: dict[DataSourceIdentity, str] | None = None,
    ) -> dict[DataSourceIdentity, UUID]:
        """Resolve each distinct identity in a batch; a batch holds only a handful of devices."""
        reported_types = reported_types or {}
        software_versions = software_versions or {}
        result: dict[DataSourceIdentity, UUID] = {}
        for identity in identities:
            user_id, device_model, source, device_id, app_id = identity
            data_source = self._resolve(
                db_session,
                provider,
                (user_id, device_model, source, *stable_ids(provider, device_id, app_id)),
                user_connection_id=user_connection_id,
                software_version=software_versions.get(identity),
                reported_type=reported_types.get(identity),
            )
            result[identity] = data_source.id
        return result

    def get_user_data_sources(
        self,
        db_session: DbSession,
        user_id: UUID,
    ) -> list[DataSource]:
        return (
            db_session.query(self.model)
            .filter(self.model.user_id == user_id)
            .order_by(asc(self.model.provider), asc(self.model.device_model))
            .all()
        )

    def delete_user_provider_data(
        self,
        db_session: DbSession,
        user_id: UUID,
        provider: ProviderName,
    ) -> int:
        """Delete all of a user's data for a single provider.

        Deletes the user's health_score rows for the provider (some are not linked to
        a data_source), then the data_source rows. ON DELETE CASCADE on the data_source
        FK removes every dependent row - event_records, data_point_series (+ archive),
        event/sleep/workout/menstrual details and health_scores linked via data_source
        or event_record. Returns the number of data_source rows deleted.
        """
        db_session.execute(
            delete(HealthScore).where(
                and_(HealthScore.user_id == user_id, HealthScore.provider == provider),
            ),
        )
        result = cast(
            CursorResult,
            db_session.execute(
                delete(self.model).where(
                    and_(self.model.user_id == user_id, self.model.provider == provider),
                ),
            ),
        )
        db_session.commit()
        return result.rowcount

    def infer_provider_from_source(self, source: str | None) -> ProviderName:
        """Infer provider from source string.

        Deprecated: Use ProviderName.from_source_string() directly instead.
        This method is kept for backward compatibility.
        """
        return ProviderName.from_source_string(source)
