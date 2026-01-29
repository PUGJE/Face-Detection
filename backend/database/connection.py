"""
Database Connection and Session Management

This module handles database connection, session creation, and initialization.

Author: Face Recognition Team
Date: January 2026
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from contextlib import contextmanager
from pathlib import Path
import logging

from backend.models.student import Base, Student, Attendance, User
from backend.config import settings

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Database Manager for SQLite
    
    Handles connection, session management, and initialization
    """
    
    def __init__(self, database_url: str = None):
        """
        Initialize database manager
        
        Args:
            database_url (str): Database URL (uses config if None)
        """
        if database_url is None:
            database_url = settings.database_url
        
        self.database_url = database_url
        self.engine = None
        self.SessionLocal = None
        
        self._initialize()
    
    def _initialize(self):
        """Initialize database engine and session factory"""
        # Create database directory if needed
        if self.database_url.startswith('sqlite:///'):
            db_path = self.database_url.replace('sqlite:///', '')
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Create engine
        # For SQLite, we use check_same_thread=False to allow multi-threading
        connect_args = {"check_same_thread": False} if 'sqlite' in self.database_url else {}
        
        self.engine = create_engine(
            self.database_url,
            connect_args=connect_args,
            echo=False  # Set to True for SQL query logging
        )
        
        # Create session factory
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
        
        logger.info(f"Database engine created: {self.database_url}")
    
    def create_tables(self):
        """Create all tables in the database"""
        Base.metadata.create_all(bind=self.engine)
        logger.info("Database tables created successfully")
    
    def drop_tables(self):
        """Drop all tables (use with caution!)"""
        Base.metadata.drop_all(bind=self.engine)
        logger.warning("All database tables dropped")
    
    def get_session(self) -> Session:
        """
        Get a new database session
        
        Returns:
            Session: SQLAlchemy session
        """
        return self.SessionLocal()
    
    @contextmanager
    def session_scope(self):
        """
        Provide a transactional scope for database operations
        
        Usage:
            with db_manager.session_scope() as session:
                session.add(student)
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            session.close()
    
    def reset_database(self):
        """Reset database (drop and recreate all tables)"""
        logger.warning("Resetting database...")
        self.drop_tables()
        self.create_tables()
        logger.info("Database reset complete")


# Global database manager instance
db_manager = DatabaseManager()


def get_db():
    """
    Dependency function for FastAPI to get database session
    
    Usage in FastAPI:
        @app.get("/students")
        def get_students(db: Session = Depends(get_db)):
            ...
    """
    db = db_manager.get_session()
    try:
        yield db
    finally:
        db.close()


def init_database():
    """Initialize database (create tables if they don't exist)"""
    db_manager.create_tables()
    logger.info("Database initialized")


# Test function
if __name__ == "__main__":
    print("=" * 60)
    print("DATABASE CONNECTION TEST")
    print("=" * 60)
    
    # Initialize database
    print("\nInitializing database...")
    init_database()
    
    # Test session
    print("\nTesting database session...")
    with db_manager.session_scope() as session:
        # Count students
        student_count = session.query(Student).count()
        print(f"Current students in database: {student_count}")
        
        # Count attendance records
        attendance_count = session.query(Attendance).count()
        print(f"Current attendance records: {attendance_count}")
    
    print("\n✓ Database connection test successful!")
    print(f"Database location: {settings.database_url}")
