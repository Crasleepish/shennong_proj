from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager
from .config import Config

engine = None
SessionLocal = None

# Create a base class for our models to inherit from
Base = declarative_base()

def init_engine(config):
    """
    Initialize the SQLAlchemy engine and session factory using the given configuration.
    This function must be called before using the database.
    """
    global engine, SessionLocal
    engine = create_engine(
        config.get("SQLALCHEMY_DATABASE_URI"),
        echo=config.get("SQLALCHEMY_ECHO")
    )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, expire_on_commit=False)


def init_db():
    """
    Import all modules that define models so that
    Base.metadata can be populated. Then create all tables.
    """
    from app.models import models  # ensure all models are imported
    from app.models import stock_models
    from app.models import task_record
    Base.metadata.create_all(bind=engine)

@contextmanager
def get_db():
    """
    Provide a transactional scope around a series of operations.
    This context manager yields a database session that automatically commits
    if no exceptions occur, or rolls back in case of errors.
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()  # Commit the transaction if no exceptions occur
    except Exception as e:
        db.rollback()  # Roll back the transaction if any error occurs
        raise e
    finally:
        db.close()  # Always close the session