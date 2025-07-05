import os
from api_key import LOCAL_DB_URI
from api_key import LOG_LEVEL, SQLALCHEMY_LOG_LEVEL

basedir = os.path.abspath(os.path.dirname(__file__))

class Config:
    # Use an environment variable if set, otherwise fall back to a default URI.
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        'DATABASE_URI',
        LOCAL_DB_URI
    )
    # Set echo to True if you want to log SQL statements (helpful in development)
    SQLALCHEMY_ECHO = False
    TESTING = False

    # Logging settings
    LOGGING_CONFIG = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'default': {
                'format': '%(asctime)s [%(levelname)s] %(name)s - %(message)s'
            },
            'detailed': {
                'format': '%(asctime)s [%(levelname)s] %(name)s - %(message)s [in %(pathname)s:%(lineno)d]'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'formatter': 'default',
                'level': LOG_LEVEL
            },
            'file': {
                'class': 'logging.handlers.TimedRotatingFileHandler',
                'filename': os.path.join(basedir, 'log', 'app.log'),
                'when': 'midnight',
                'backupCount': 7,
                'encoding': 'utf-8',
                'formatter': 'default',
                'level': LOG_LEVEL
            }
        },
        'root': {
            'handlers': ['console', 'file'],
            'level': LOG_LEVEL
        },
        'loggers': {
            # For SQLAlchemy engine logging, set a higher log level to avoid verbose output.
            'sqlalchemy.engine': {
                'handlers': ['console', 'file'],
                'level': SQLALCHEMY_LOG_LEVEL,
                'propagate': False
            }
        }
    }

class TestConfig(Config):
    TESTING = True
    # Use an in-memory SQLite database for tests.
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        'DATABASE_URI',
        LOCAL_DB_URI
    )
    SQLALCHEMY_ECHO = True
    LOGGING_CONFIG = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'default': {
                'format': '%(asctime)s [%(levelname)s] %(name)s - %(message)s'
            },
            'detailed': {
                'format': '%(asctime)s [%(levelname)s] %(name)s - %(message)s [in %(pathname)s:%(lineno)d]'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'formatter': 'default',
                'level': 'DEBUG'
            }
        },
        'root': {
            'handlers': ['console'],
            'level': 'DEBUG'
        },
        'loggers': {
            # For SQLAlchemy engine logging, set a higher log level to avoid verbose output.
            'sqlalchemy.engine': {
                'handlers': ['console'],
                'level': 'INFO',
                'propagate': False
            }
        }
    }
