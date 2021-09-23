import os
import json
import shutil
import logging
import tempfile

from flask import Blueprint
from flask import current_app
import celery

from flask_jwt_extended import get_jwt_identity
from flask_jwt_extended import jwt_required

from flask_restful import Api, Resource, abort

from flask_apispec.extension import FlaskApiSpec
from flask_apispec.views import MethodResource
from flask_apispec import marshal_with, doc, use_kwargs

from modep_common.models import (
    db,
    User,
    TabularDataset,
    TabularFramework,
    TabularFrameworkService,
    TabularFrameworkPredictions,
)

from modep_common.schemas import (
    TabularFrameworkSchema,
    TabularFrameworkParamsSchema,
    TabularFrameworkPredictSchema,
    TabularFrameworkPredictionsSchema,
)

from modep_common.io import StorageClient

from modep_amlb import tasks


BLUEPRINT = 'api'
TAG = 'Frameworks'

blueprint = Blueprint(BLUEPRINT, __name__)
api = Api(blueprint)
docs = FlaskApiSpec()

logger = logging.getLogger(__name__)

# A benchmark definition consists of a datasets definition and a constraints definition.
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
  task_type: {task_type}
  model_path: {model_path}
"""

constraint_template = """---
constraint:
  cores: 8
  max_runtime_seconds: {max_runtime_seconds}
"""


def tabular_dataset_by_id(id):
    objs = TabularDataset.query.filter_by(id=id)
    if objs.count() == 0:
        abort(500, message=f"No dataset found with the id: '{id}'")
    elif objs.count() == 1:
        return objs[0]
    elif objs.count() > 1:
        abort(500, message=f"Multiple datasets found with the id: '{id}'")


def yaml_path_string(paths):
    """ Create yaml for a list of paths """
    out = ''
    for i, p in enumerate(paths):
        out += '      - %s' % p
        if i < len(paths) - 1:
            out += '\n'
    return out


def write_string(string, name, outdir):
    with open(os.path.join(outdir, name), 'w+') as f:
        f.write(string)


class TabularFrameworkTrain(MethodResource, Resource):

    @doc(description='Run a framework', tags=[TAG])
    @use_kwargs(TabularFrameworkParamsSchema, location=('json'))
    @marshal_with(TabularFrameworkSchema)
    @jwt_required()
    def post(self, **kwargs):

        user = User.query.filter_by(api_key=get_jwt_identity()).one()
        user_pk = user.pk
        n_folds = len(kwargs['train_ids'])

        framework_name = kwargs['framework_name']
        # framework_id is the lower cased framework name
        framework_svc = TabularFrameworkService.query.filter_by(framework_id=framework_name.lower())
        count = framework_svc.count()
        if count == 0:
            names = [x.framework_name for x in TabularFrameworkService.query.all()]
            logger.info(names)
            names = ', '.join(names)
            abort(404, message=f"Unknown framework name: {framework_name}, should be one of: {names}")
        elif count > 1:
            abort(404, message=f"Multiple frameworks with name: {framework_name}")
        framework_svc = framework_svc[0]

        # add things for creating TabularFramework object
        kwargs['user_pk'] = user_pk
        kwargs['n_folds'] = n_folds
        kwargs['framework_name'] = framework_name
        kwargs['framework_pk'] = framework_svc.pk

        framework = TabularFramework(**kwargs)
        framework.status = 'RUNNING'

        outdir = tempfile.NamedTemporaryFile().name
        framework.outdir = outdir
        logger.info('Saving output to %s', outdir)

        db.session.add(framework)
        db.session.commit()

        # Fetch this here and use it instead of framework.pk due to DB session
        # timeout in SQLAlchemy when lazily getting object properties.
        framework_pk = framework.pk

        if os.path.exists(outdir):
            shutil.rmtree(outdir)
        os.makedirs(outdir, exist_ok=True)

        for d in ['input-data/train', 'input-data/test', 'benchmarks']:
            os.makedirs(os.path.join(outdir, d), exist_ok=True)

        sc = StorageClient()

        train = []
        for i, id in enumerate(kwargs['train_ids']):
            dset = tabular_dataset_by_id(id)
            dest_path = os.path.join(outdir, 'input-data', 'train', f'{dset.id}_{i}.{dset.ext}')
            sc.download(dset.gcp_path, dest_path)
            train.append(dest_path)

        test = []
        for i, id in enumerate(kwargs['test_ids']):
            dset = tabular_dataset_by_id(id)
            dest_path = os.path.join(outdir, 'input-data', 'test', f'{dset.id}_{i}.{dset.ext}')
            sc.download(dset.gcp_path, dest_path)
            test.append(dest_path)

        config = config_template.format(
            outdir=outdir,
        )
        benchmark = benchmark_template.format(
            train=yaml_path_string(train),
            test=yaml_path_string(test),
            target=kwargs['target'],
            n_folds=n_folds,
            max_runtime_seconds=kwargs['max_runtime_seconds'],
            task_type='train', # default task
            model_path='', # no existing model
        )
        constraint = constraint_template.format(
            max_runtime_seconds=kwargs['max_runtime_seconds'],
        )

        # write all yaml files
        write_string(config, 'config.yaml', outdir)
        write_string(benchmark, 'benchmarks/benchmark.yaml', outdir)
        write_string(constraint, 'constraints.yaml', outdir)

        logger.info(config)
        logger.info(benchmark)
        logger.info(constraint)

        # link gets run if first task is successful
        # link_error gets run if first task fails
        result = current_app.celery.send_task(
            'tasks.runbenchmark',
            args=(framework_pk, framework_name, 'benchmark', 'constraint', outdir, outdir),
            link=[celery.signature('tasks.runbenchmark_on_success', args=(framework_pk, outdir))],
            link_error=[celery.signature('tasks.runbenchmark_on_failure', args=(framework_pk, outdir))],
        )
        logger.info('celery send_task result: %s', str(result))

        # Re-query because it could have been a long time since creating the object
        # was created above before training and SQL session could be detached.
        framework = TabularFramework.query.filter_by(pk=framework_pk).one()

        return framework


class TabularFrameworkPredict(MethodResource, Resource):

    @doc(description='Get predictions from a trained framework', tags=[TAG])
    @use_kwargs(TabularFrameworkPredictSchema, location=('json'))
    @marshal_with(TabularFrameworkPredictionsSchema)
    @jwt_required()
    def post(self, **kwargs):

        logger.info(f"kwargs: {kwargs}")

        user = User.query.filter_by(api_key=get_jwt_identity()).one()
        user_pk = user.pk

        framework = TabularFramework.query.filter_by(id=kwargs['framework_id']).one()
        # Fetch this here and use it instead of framework.pk due to DB session
        # timeout in SQLAlchemy when lazily getting object properties.
        framework_pk = framework.pk
        framework_name = framework.framework_name

        # TODO: default to the fold used in training for this dataset
        model_fold = 0

        # TODO: set to users maximum allowed training time
        max_runtime_seconds = 3600 # kwargs['max_runtime_seconds']

        outdir = tempfile.NamedTemporaryFile().name
        logger.info('Saving output to %s', outdir)

        if os.path.exists(outdir):
            shutil.rmtree(outdir)
        os.makedirs(outdir, exist_ok=True)

        for d in ['input-data/train', 'input-data/test', 'benchmarks']:
            os.makedirs(os.path.join(outdir, d), exist_ok=True)

        sc = StorageClient()

        # TODO: training IDs aren't used but we need to have something
        train_ids = json.loads(framework.train_ids)
        train = []
        for i, id in enumerate(train_ids):
            dset = tabular_dataset_by_id(id)
            dest_path = os.path.join(outdir, 'input-data', 'train', f'{dset.id}_{i}.{dset.ext}')
            sc.download(dset.gcp_path, dest_path)
            train.append(dest_path)

        dset = tabular_dataset_by_id(kwargs['dataset_id'])
        dest_path = os.path.join(outdir, 'input-data', 'test', f'{dset.id}_{0}.{dset.ext}')
        sc.download(dset.gcp_path, dest_path)
        test = [dest_path]

        preds = TabularFrameworkPredictions(framework_pk, dset.pk, model_fold)
        preds.status = 'RUNNING'
        db.session.add(preds)

        # to get pred_pk
        db.session.commit()
        preds_pk = preds.pk

        # download the model
        model_path = os.path.join(outdir, 'input-data', 'saved-model.zip')
        gcp_model_path = framework.gcp_model_paths[model_fold]
        if gcp_model_path is None:
            # case when model saving failed
            # TODO: catch this earlier
            preds.status = 'FAIL'
            db.session.add(preds)
            db.session.commit()
            return preds

        sc.download(gcp_model_path, model_path)
        # extract to path without zip in it
        shutil.unpack_archive(model_path, model_path.replace('.zip', ''), 'zip')
        # remove extension so that path points to extracted zip contents
        model_path = model_path.replace('.zip', '')

        config = config_template.format(
            outdir=outdir,
        )
        benchmark = benchmark_template.format(
            train=yaml_path_string(train),
            test=yaml_path_string(test),
            target=framework.target,
            n_folds=1,
            max_runtime_seconds=max_runtime_seconds,
            task_type='predict',
            model_path=model_path,
        )
        constraint = constraint_template.format(
            max_runtime_seconds=max_runtime_seconds,
        )

        # write all yaml files
        write_string(config, 'config.yaml', outdir)
        write_string(benchmark, 'benchmarks/benchmark.yaml', outdir)
        write_string(constraint, 'constraints.yaml', outdir)

        logger.info(config)
        logger.info(benchmark)
        logger.info(constraint)

        result = current_app.celery.send_task(
            'tasks.runbenchmark_predict',
            args=(framework_pk, framework_name, 'benchmark', 'constraint', outdir, outdir),
            link=[celery.signature('tasks.runbenchmark_predict_on_success', args=(outdir, preds_pk,))],
            link_error=[celery.signature('tasks.runbenchmark_predict_on_failure', args=(preds_pk,))],
        )
        logger.info('celery send_task result: %s', str(result))

        # Re-query because it could have been a long time since creating the object
        # was created above before training and SQL session could be detached.
        preds = TabularFrameworkPredictions.query.filter_by(pk=preds_pk).one()

        return preds

api.add_resource(TabularFrameworkTrain, "/frameworks/tabular")
docs.register(TabularFrameworkTrain, blueprint=BLUEPRINT)

api.add_resource(TabularFrameworkPredict, "/frameworks/tabular/predict")
docs.register(TabularFrameworkPredict, blueprint=BLUEPRINT)

# class TabularFrameworkUpdater(MethodResource, Resource):
#     """ Just for running a big batch job to update all framework runs. Should NOT be exposed """

#     @doc(description='Update results for all frameworks', tags=[TAG])
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
# docs.register(TabularFrameworkUpdater, blueprint=BLUEPRINT)
