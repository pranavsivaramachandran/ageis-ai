from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from aegis.core.config import config
from aegis.core.logger import get_logger
from aegis.db.models.base import Base

logger = get_logger(__name__)

# Create the SQLAlchemy engine
# SQLite is used for the foundation phase to ensure determinism and easy testing
engine = create_engine(
    config.DATABASE_URL,
    connect_args={"check_same_thread": False} if config.DATABASE_URL.startswith("sqlite") else {}
)

# Create a configured "Session" class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    """Initializes the database schema."""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database schema initialized")
    except Exception as e:
        logger.exception("Failed to initialize database schema", exc_info=e)
        raise

@contextmanager
def get_db() -> Session:
    """
    Context manager that provides a database session.
    Provides a transactional scope around a series of operations.
    Usage: with get_db() as session: ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

