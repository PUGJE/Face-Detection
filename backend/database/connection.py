"""
Database Connection and Session Management

Connects to a PostgreSQL database (Supabase) via SQLAlchemy.
The DATABASE_URL environment variable must be set to a valid
PostgreSQL connection string, e.g.:
  postgresql://postgres:[PASSWORD]@db.[PROJECT].supabase.co:5432/postgres

All public symbols are unchanged from the SQLite version so no
other module requires any edits:
  - db_manager   : DatabaseManager instance
  - get_db()     : FastAPI dependency that yields a Session
  - init_database(): creates tables on first run

Author: Face Recognition Team
"""

import logging
from contextlib import contextmanager

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session

from backend.models.student import Base, Student, Attendance, User
from backend.config import settings

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Database Manager for PostgreSQL (Supabase)

    Handles connection pooling, session management, and schema
    initialisation.  Uses the DATABASE_URL from settings/env.
    """

    def __init__(self, database_url: str = None):
        if database_url is None:
            database_url = settings.database_url

        self.database_url = database_url
        self.engine = None
        self.SessionLocal = None

        self._initialize()

    def _initialize(self):
        """Create the SQLAlchemy engine with PostgreSQL-tuned pool settings."""
        self.engine = create_engine(
            self.database_url,
            # Connection pool settings — sensible defaults for a hosted DB
            pool_size=5,          # keep 5 connections warm
            max_overflow=10,      # allow up to 10 extra connections under load
            pool_timeout=30,      # wait up to 30s for a connection from the pool
            pool_recycle=1800,    # recycle connections every 30 min (avoids stale conns)
            pool_pre_ping=True,   # test connection health before using from pool
            echo=False,           # set True to log all SQL queries
        )

        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine,
        )

        logger.info(f"Database engine created: {self._safe_url()}")

    def _safe_url(self) -> str:
        """Return the database URL with the password redacted for logging."""
        try:
            from urllib.parse import urlparse, urlunparse
            parsed = urlparse(self.database_url)
            redacted = parsed._replace(netloc=parsed.netloc.replace(
                parsed.password or "", "****"
            ))
            return urlunparse(redacted)
        except Exception:
            return "<database_url>"

    # ------------------------------------------------------------------
    # Table management
    # ------------------------------------------------------------------

    def create_tables(self):
        """Create all ORM-defined tables if they don't already exist."""
        Base.metadata.create_all(bind=self.engine)
        logger.info("Database tables verified / created.")

    def drop_tables(self):
        """Drop all tables — use with extreme caution!"""
        Base.metadata.drop_all(bind=self.engine)
        logger.warning("All database tables dropped.")

    def reset_database(self):
        """Drop and recreate all tables (destructive!)."""
        logger.warning("Resetting database …")
        self.drop_tables()
        self.create_tables()
        logger.info("Database reset complete.")

    # ------------------------------------------------------------------
    # Session helpers
    # ------------------------------------------------------------------

    def get_session(self) -> Session:
        """Return a new SQLAlchemy session (caller is responsible for closing)."""
        return self.SessionLocal()

    @contextmanager
    def session_scope(self):
        """
        Transactional context manager.

        Usage:
            with db_manager.session_scope() as session:
                session.add(obj)
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as exc:
            session.rollback()
            logger.error(f"Database error (rolled back): {exc}")
            raise
        finally:
            session.close()

    def health_check(self) -> bool:
        """Return True if the database is reachable."""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception as exc:
            logger.error(f"Database health check failed: {exc}")
            return False


# ---------------------------------------------------------------------------
# Global singleton — imported by services, routes, and attendance_system
# ---------------------------------------------------------------------------
db_manager = DatabaseManager()


def get_db():
    """
    FastAPI dependency — yields a Session and closes it after the request.

    Usage:
        @router.get("/students")
        def list_students(db: Session = Depends(get_db)):
            ...
    """
    db = db_manager.get_session()
    try:
        yield db
    finally:
        db.close()


def init_database():
    """Called at application startup to ensure all tables exist."""
    db_manager.create_tables()
    logger.info("Database initialised.")


# ---------------------------------------------------------------------------
# Manual smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("DATABASE CONNECTION TEST")
    print("=" * 60)

    print("\nChecking connectivity …")
    ok = db_manager.health_check()
    print(f"  Health check: {'✓ OK' if ok else '✗ FAILED'}")

    if ok:
        print("\nInitialising tables …")
        init_database()

        with db_manager.session_scope() as s:
            students = s.query(Student).count()
            records  = s.query(Attendance).count()
            print(f"  Students : {students}")
            print(f"  Attendance records: {records}")

        print(f"\n✓ Connected to: {db_manager._safe_url()}")
