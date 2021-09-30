import os
import logging
import datetime
import secrets
from flask import Flask
import flask_login
# from flask_sqlalchemy import SQLAlchemy
from flask_jwt_extended import JWTManager

from modep_common.models import db
from modep_common import settings

from modep_amlb.v1.api_def import blueprint as blueprint_v1, docs as docs_v1

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.DEBUG,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

app = Flask(__name__)

app.logger.setLevel(logging.DEBUG)
app.config['SQLALCHEMY_DATABASE_URI'] = settings.SQLALCHEMY_DATABASE_URI
app.config['SQLALCHEMY_TRACK_MODIFICATION'] = False
app.config['JSON_SORT_KEYS'] = False
app.config['EXPLAIN_TEMPLATE_LOADING'] = False
app.config['SECRET_KEY'] = secrets.token_urlsafe(24)
app.config["JWT_SECRET_KEY"] = os.environ.get('JWT_SECRET_KEY', secrets.token_urlsafe(24))
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = datetime.timedelta(hours=2)

app.register_blueprint(blueprint_v1, url_prefix="/v1")

## Next line commented out due to error inside marshmallow (we don't need docs for this service anyway)
# docs_v1.init_app(app)

## Initialize SQLAlchemy DB
db.init_app(app)

## don't need migrations b/c they're handled by flask_app
# migrate = Migrate(app, db)

login_manager = flask_login.LoginManager()
login_manager.init_app(app)

jwt = JWTManager(app)

from celery import Celery

def make_celery(app):
    celery = Celery(
        app.import_name,
        backend=app.config['CELERY_RESULT_BACKEND'],
        broker=app.config['CELERY_BROKER_URL'],
        include=['modep_amlb.tasks'],
    )
    celery.conf.update(app.config)

    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)

    celery.Task = ContextTask
    return celery

app.config.update(
    CELERY_BROKER_URL='redis://localhost:6379',
    CELERY_RESULT_BACKEND='redis://localhost:6379'
)
celery = make_celery(app)
app.celery = celery
