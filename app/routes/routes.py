from flask import Blueprint, request, jsonify
from app.database import get_db
from app.dao.crud import create_user, get_users, get_user_by_username
import logging

bp = Blueprint('users', __name__, url_prefix='/users')
logger = logging.getLogger(__name__)  # Get a logger for this module

@bp.route('/', methods=['GET'])
def index():
    return b"Hello, Flask with PostgreSQL and SQLAlchemy!", 200

@bp.route('/list', methods=['GET'])
def list_users():
    logger.debug("calling - /list")
    # Use the context manager to obtain a session
    with get_db() as db:
        users = get_users(db)
        results = [{'id': user.id, 'username': user.username, 'email': user.email} for user in users]
        return jsonify(results), 200

@bp.route('/add', methods=['POST'])
def add_user():
    data = request.get_json()
    username = data.get('username')
    email = data.get('email')
    if not username or not email:
        logger.warning("Missing username or email in the request data.")
        return jsonify({'error': 'Missing username or email'}), 400
    with get_db() as db:
        if get_user_by_username(db, username=username):
            logger.warning("User with username '%s' already exists.", username)
            return jsonify({'error': 'User with that username already exists.'}), 409

        user = create_user(db, username=username, email=email)
        return jsonify({'id': user.id, 'username': user.username, 'email': user.email}), 201
