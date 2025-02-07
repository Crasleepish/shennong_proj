from flask import Flask
from .config import Config
from .database import init_engine, init_db
from .routes.routes import bp as users_bp
import logging
import logging.config

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Configure logging using the settings in app.config.
    logging.config.dictConfig(app.config.get('LOGGING_CONFIG'))
    logger = logging.getLogger(__name__)
    logger.debug("Logging is configured.")

    # Initialize the engine using the current configuration.
    init_engine(app.config)

    # Initialize the database (create tables if they don't exist)
    init_db()
    
    # Register blueprints for modular route definitions
    app.register_blueprint(users_bp)
    
    return app
