import logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.DEBUG,
)

import os
import datetime
from flask import Flask
import flask_login
from flask_sqlalchemy import SQLAlchemy
## handled by main `flask_app`
# from flask_migrate import Migrate
from flask_jwt_extended import JWTManager

from app_utils.models import db
from app_utils import settings

from modep_amlb.v1.api_def import blueprint as blueprint_v1, docs as docs_v1

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# create the uploads folder if it doesn't already exist
os.makedirs(settings.UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)

app.logger.setLevel(logging.DEBUG)
app.config['UPLOAD_FOLDER'] = settings.UPLOAD_FOLDER
app.config['SQLALCHEMY_DATABASE_URI'] = settings.SQLALCHEMY_DATABASE_URI
app.config['SQLALCHEMY_TRACK_MODIFICATION'] = True
app.config['JSON_SORT_KEYS'] = False
app.config['EXPLAIN_TEMPLATE_LOADING'] = False
app.config['SECRET_KEY'] = 'iVrGj21Ps8S9YeyIorg9iMbIKDSEHRE5'
app.config["JWT_SECRET_KEY"] = 'iVrGj21Ps8S9YeyIorg9iMbIKDSEHRE5'
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = datetime.timedelta(hours=2)

app.register_blueprint(blueprint_v1, url_prefix="/v1")

'''
Next line commented out due to error inside marshmallow (we don't need docs for this service anyway)
  File "/bench/venv/lib/python3.7/site-packages/apispec/ext/marshmallow/field_converter.py", line 219, in field2default
    default = field.load_default
AttributeError: 'String' object has no attribute 'load_default'
'''
# docs_v1.init_app(app)

# db = SQLAlchemy(app)
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

# @celery.task()
# def add_together(a, b):
#     print('Adding', a, b)
#     return a + b

app.celery = celery
