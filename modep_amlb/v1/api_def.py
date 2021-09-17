import os
import shutil
import logging
import tempfile

import pandas as pd

from flask import Blueprint
from flask import current_app
import celery

from flask_jwt_extended import get_jwt_identity
from flask_jwt_extended import jwt_required

from flask_restful import Api, Resource, abort

from flask_apispec.extension import FlaskApiSpec
from flask_apispec.views import MethodResource
from flask_apispec import marshal_with, doc, use_kwargs

from modep_amlb import settings
from modep_amlb import tasks

from app_utils.models import (
    db,
    AnonUser,
    User,
    TabularDataset,
    TabularFramework,
    TabularFrameworkService,
)

from app_utils.schemas import (
    TabularFrameworkSchema,
    TabularFrameworkParamsSchema,
)

from app_utils.io import StorageClient

blueprint_name = 'api'
blueprint = Blueprint(blueprint_name, __name__)
api = Api(blueprint)
docs = FlaskApiSpec()

logger = logging.getLogger(__name__)

tag = 'Frameworks'

config_template = """---
benchmarks:
  definition_dir:
    - '{outdir}/benchmarks'
  constraints_file:
    - '{outdir}/constraints.yaml'
"""

benchmark_template = """---
- name: files
  dataset:
    train:
{train}
    test:
{test}
    target: {target}
  folds: {n_folds}
"""

constraint_template = """---
constraint:
  cores: 8
  max_runtime_seconds: {max_runtime_seconds}
"""


class TabularFrameworkList(MethodResource, Resource):

    @doc(description='Run a framework', tags=[tag])
    @use_kwargs(TabularFrameworkParamsSchema, location=('json'))
    @marshal_with(TabularFrameworkSchema)
    @jwt_required()
    def post(self, **kwargs):

        user = User.query.filter_by(api_key=get_jwt_identity()).one()
        user_pk = user.pk
        kwargs['user_pk'] = user_pk

        # TODO: can train/test have a different number of folds?
        n_folds = len(kwargs['train_ids'])
        kwargs['n_folds'] = n_folds

        framework_id = kwargs['framework_id']
        framework_service = TabularFrameworkService.query.filter_by(framework_id=framework_id).one()
        kwargs['framework_name'] = framework_service.framework_name

        framework = TabularFramework(**kwargs)
        framework.status = 'RUNNING'
        db.session.add(framework)
        db.session.commit()

        # Fetch this here and use it instead of framework.pk due to DB session
        # timeout in SQLAlchemy when lazily getting object properties.
        framework_pk = framework.pk

        outdir = tempfile.NamedTemporaryFile().name
        logger.info('Saving output to %s', outdir)

        if os.path.exists(outdir):
            shutil.rmtree(outdir)
        os.makedirs(outdir, exist_ok=True)

        for d in ['input-data/train', 'input-data/test', 'benchmarks']:
            os.makedirs(os.path.join(outdir, d), exist_ok=True)

        sc = StorageClient()

        def tabular_dataset_by_name(name):
            # different users can have one dataset with the same name
            objs = TabularDataset.query.filter_by(name=name, user_pk=user_pk)
            if objs.count() == 0:
                abort(500, message=f"No dataset found with the name: '{name}'")
            elif objs.count() == 1:
                return objs[0]
            elif objs.count() > 1:
                abort(500, message=f"Multiple datasets found with the name: '{name}'")

        train = []
        for i, id in enumerate(kwargs['train_ids']):
            # dset = TabularDataset.query.filter_by(id=id).one()
            dset = tabular_dataset_by_name(id)
            dest_path = os.path.join(outdir, 'input-data', 'train', '%s_%i.%s' % (dset.id, i, dset.ext))
            sc.download(dset.gcp_path, dest_path)
            train.append(dest_path)

        test = []
        for i, id in enumerate(kwargs['test_ids']):
            # dset = TabularDataset.query.filter_by(id=id).one()
            dset = tabular_dataset_by_name(id)
            dest_path = os.path.join(outdir, 'input-data', 'test', '%s_%i.%s' % (dset.id, i, dset.ext))
            sc.download(dset.gcp_path, dest_path)
            test.append(dest_path)

        def make_items(paths):
            out = ''
            for i, p in enumerate(paths):
                out += '      - %s' % p
                if i < len(paths) - 1:
                    out += '\n'
            return out

        train, test = make_items(train), make_items(test)

        def write(string, name):
            with open(os.path.join(outdir, name), 'w+') as f:
                f.write(string)

        config = config_template.format(outdir=outdir)
        benchmark = benchmark_template.format(train=train, test=test, target=kwargs['target'],
                                              n_folds=n_folds,
                                              max_runtime_seconds=kwargs['max_runtime_seconds'])
        constraint = constraint_template.format(max_runtime_seconds=kwargs['max_runtime_seconds'])
        write(config, 'config.yaml')
        write(benchmark, 'benchmarks/benchmark.yaml')
        write(constraint, 'constraints.yaml')

        logger.info(config)
        logger.info(benchmark)
        logger.info(constraint)

        USE_CELERY = int(os.environ.get('USE_CELERY', '1'))
        logger.info('USE_CELERY: %i', USE_CELERY)

        if USE_CELERY:
            # link gets run if first task is successful
            # link_error gets run if first task fails
            result = current_app.celery.send_task(
                'tasks.run_benchmark',
                args=(framework_pk, kwargs['framework_id'], 'benchmark', 'constraint', outdir, outdir),
                link=[celery.signature('tasks.on_success', args=(framework_pk, outdir))],
                link_error=[celery.signature('tasks.on_failure', args=(framework_pk, outdir))],
            )
            logger.info('celery send_task result: %s', str(result))
        else:
            tasks.run_benchmark(framework_pk, kwargs['framework_id'], 'benchmark', 'constraint', outdir, outdir)
            tasks.on_success(None, framework_pk, outdir)

        # Re-query because it could have been a long time since creating the object
        # was created above before training and SQL session could be detached.
        framework = TabularFramework.query.filter_by(pk=framework_pk).one()

        return framework


api.add_resource(TabularFrameworkList, f"/frameworks/tabular")

docs.register(TabularFrameworkList, blueprint=blueprint_name)


# class TabularFrameworkUpdater(MethodResource, Resource):
#     """ Just for running a big batch job to update all framework runs. Should NOT be exposed """

#     @doc(description='Update results for all frameworks', tags=[tag])
#     # @jwt_required()
#     def post(self, **kwargs):
#         frameworks = TabularFramework.query.all()

#         sc = StorageClient()

#         for framework in frameworks:
#             # if framework.status != 'SUCCESS':
#             #     continue

#             if os.path.exists(framework.outdir):
#                 shutil.rmtree(framework.outdir)
#             if os.path.exists(framework.outdir + '.zip'):
#                 os.remove(framework.outdir + '.zip')

#             sc.download(framework.gcp_path, framework.outdir + '.zip')
#             shutil.unpack_archive(framework.outdir + '.zip', framework.outdir, 'zip')

#             tasks.on_success(None, framework.pk, framework.outdir, upload=False)
# api.add_resource(TabularFrameworkUpdater, f"/frameworks/tabular/update-results")
# docs.register(TabularFrameworkUpdater, blueprint=blueprint_name)
