from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, String, Text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from medcode_rag.config import settings


class Base(DeclarativeBase):
    pass


class EncodingAuditLog(Base):
    __tablename__ = "encoding_audit_log"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    transcript_id: Mapped[str] = mapped_column(String(128), index=True)
    entries: Mapped[dict] = mapped_column(JSON)
    flagged_for_review: Mapped[bool]
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


class FeedbackAuditLog(Base):
    __tablename__ = "feedback_audit_log"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    transcript_id: Mapped[str] = mapped_column(String(128), index=True)
    original_code: Mapped[str] = mapped_column(String(32))
    corrected_code: Mapped[str] = mapped_column(String(32))
    reason: Mapped[str] = mapped_column(Text)
    context_snippet: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


engine = create_async_engine(settings.postgres_dsn, pool_pre_ping=True, pool_size=10)
async_session = async_sessionmaker(engine, expire_on_commit=False)


async def init_db() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_session() -> AsyncSession:
    async with async_session() as session:
        yield session
