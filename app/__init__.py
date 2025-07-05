from flask import Flask
from flask_httpauth import HTTPBasicAuth
from werkzeug.security import check_password_hash, generate_password_hash
from .config import Config
from .database import init_engine, init_db
from .routes.routes import bp as users_bp
from .routes.fin_data_routes import fin_data_bp
from .routes.service_routes import service_bp
from .routes.task_routes import task_bp
import logging
import logging.config
import gzip
import shutil
import os
from logging.handlers import TimedRotatingFileHandler
from api_key import API_AUTH_USER, API_AUTH_PASSWORD

auth = HTTPBasicAuth()

# 用户名和密码（可替换为数据库或配置文件中加载）
users = {
    API_AUTH_USER: generate_password_hash(API_AUTH_PASSWORD),
}

@auth.verify_password
def verify_password(username, password):
    if username in users and check_password_hash(users.get(username), password):
        return username
    return None

def create_app(config_class=Config):
    app = Flask(__name__)

    @app.before_request
    @auth.login_required
    def require_auth():
        pass

    app.config.from_object(config_class)

    # Configure logging using the settings in app.config.
    logging.config.dictConfig(app.config.get('LOGGING_CONFIG'))
    # 获取 file handler 并设置 gzip 压缩
    for handler in logging.getLogger().handlers:
        if isinstance(handler, TimedRotatingFileHandler):
            # 压缩旧文件为 .gz
            handler.namer = lambda name: name + ".gz"
            def rotator(source, dest):
                with open(source, "rb") as sf, gzip.open(dest, "wb") as df:
                    shutil.copyfileobj(sf, df)
                os.remove(source)
            handler.rotator = rotator
    logger = logging.getLogger(__name__)
    logger.debug("Logging is configured.")

    # Initialize the engine using the current configuration.
    init_engine(app.config)

    # Initialize the database (create tables if they don't exist)
    init_db()

    # Register blueprints for modular route definitions
    app.register_blueprint(users_bp)
    app.register_blueprint(fin_data_bp)
    app.register_blueprint(task_bp)
    app.register_blueprint(service_bp)
    
    return app
