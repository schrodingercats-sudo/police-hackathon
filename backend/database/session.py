"""Database Session & Engine Management for Indian Police Stolen Vehicle AI System."""

import logging
import sys
from pathlib import Path
from typing import Generator, Optional

# Ensure project root is in sys.path
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session

from backend.config import settings
from backend.database.models import Base

logger = logging.getLogger(__name__)


def create_db_engine(db_url: Optional[str] = None, echo: Optional[bool] = None) -> Engine:
    """
    Factory function to create a SQLAlchemy engine with appropriate dialect configurations.
    Enforces SQLite foreign key constraints and optimizes PostgreSQL connection pools.
    """
    target_url = db_url or settings.DATABASE_URL
    target_echo = echo if echo is not None else settings.DB_ECHO

    if target_url.startswith("sqlite"):
        engine = create_engine(
            target_url,
            connect_args={"check_same_thread": False},
            echo=target_echo,
        )

        # Enable foreign key constraint enforcement in SQLite
        @event.listens_for(engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    else:
        # Production connection pool settings for PostgreSQL / MySQL
        engine = create_engine(
            target_url,
            echo=target_echo,
            pool_pre_ping=True,
            pool_size=10,
            max_overflow=20,
        )

    return engine


# Default global engine and sessionmaker
engine = create_db_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency that provides an isolated database session per request,
    ensuring cleanup and closure in a finally block.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db(engine_override: Optional[Engine] = None) -> None:
    """
    Initialize all database tables defined in SQLAlchemy ORM models.
    """
    active_engine = engine_override or engine
    logger.info(f"Initializing database schema on {active_engine.url}...")
    Base.metadata.create_all(bind=active_engine)
    logger.info("Database schema initialized successfully.")


def reset_db(engine_override: Optional[Engine] = None) -> None:
    """
    Drops and recreates all tables. Used primarily for integration testing and clean resets.
    """
    active_engine = engine_override or engine
    logger.warning(f"Resetting database tables on {active_engine.url}...")
    Base.metadata.drop_all(bind=active_engine)
    Base.metadata.create_all(bind=active_engine)
    logger.info("Database reset completed successfully.")
