import os
import json
import glob
import shutil
import logging
import subprocess
import numpy as np
import pandas as pd

from celery import shared_task

from modep_common.models import (
    db,
    User,
    TabularDataset,
    TabularFramework,
    TabularFrameworkPredictions,
)
from modep_common.io import StorageClient

logger = logging.getLogger(__name__)


def run_cmd(cmd):
    """ Run a shell command and print the output as it runs. """
    logger.debug('$ %s', cmd)
    process = subprocess.Popen(cmd,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT,
                               universal_newlines=True,
                               shell=True)
    return_code = None
    while True:
        output = process.stdout.readline()
        return_code = process.poll()
        if return_code is not None:
            logger.debug('RETURN CODE: %i', return_code)
            logger.debug('RETURN CMD:  %s', cmd)
            for output in process.stdout.readlines():
                logger.debug(output.strip())
            break
        logger.debug(output.strip())
    return return_code


def is_local():
    """ Returns true if running locally """
    if os.path.exists('/bench/venv/bin'):
        # running in docker container
        return False
    else:
        # running locally
        return True


def get_runbenchmark_cmd():
    """ Returns the path to the python command """
    if is_local():
        return 'python -W ignore /home/jimmie/git/mlapi/automlbenchmark/runbenchmark.py'
    else:
        return '/bench/venv/bin/python3 -W ignore /bench/automlbenchmark/runbenchmark.py'


def zip_and_upload(outdir, gcp_path):
    # exclude input data since it's already been uploaded
    data_dir = os.path.join(outdir, 'input-data')
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)

    # create zip file of output
    shutil.make_archive(outdir, 'zip', outdir)

    # upload
    sc = StorageClient()
    sc.upload(outdir + '.zip', gcp_path)


def remove_files(outdir):
    """ Remove the files after zip_and_upload """
    if is_local():
        # skip this if running locally so that debugging is easier
        return

    if os.path.exists(outdir):
        logger.info("Deleting outdir to save space: %s", outdir)
        shutil.rmtree(outdir)

    zip_file = outdir + '.zip'
    if os.path.exists(zip_file):
        logger.info("Deleting outdir zip to save space: %s", zip_file)
        os.remove(zip_file)


@shared_task(name='tasks.runbenchmark', bind=True)
def runbenchmark(self, framework_pk, framework_name, dataset_id, constraint_id, outdir, user_dir):
    """
    Calls the `runbenchmark.py` script.
    """
    logger.info('self.request: %s', str(vars(self.request)))

    ## update things that should be set before running
    framework = TabularFramework.query.filter_by(pk=framework_pk).one()
    framework.outdir = outdir
    # get and save the celery task id
    framework.task_id = self.request.id
    db.session.add(framework)
    db.session.commit()

    runbenchmark_cmd = get_runbenchmark_cmd()
    cmd = f"{runbenchmark_cmd} {framework_name} {dataset_id} {constraint_id} -o {outdir} -u {user_dir} --logging debug"
    exit_code = run_cmd(cmd)
    if exit_code != 0:
        raise Exception(f"Command exited with non-zero exit status: {exit_code}")
    return exit_code


@shared_task(name='tasks.runbenchmark_on_success')
def runbenchmark_on_success(prev_result, framework_pk, outdir):
    """
    Called when `runbenchmark` runs successfullly.
    prev_result is not used (required to be passed b/c of the task linkage with `runbenchmark`)
    """

    framework = TabularFramework.query.filter_by(pk=framework_pk).one()
    user = User.query.filter_by(pk=framework.user_pk).one()

    # upload results to GCP bucket
    # gcp_path = f"tabular-frameworks/{user.id}/{framework.id}.zip"
    # zip_and_upload(outdir, gcp_path)
    # framework.gcp_path = gcp_path

    #-----------------------------------------------------
    # metadata.json (one file per fold)

    fs = sorted(glob.glob(outdir + '/*/predictions/files/*/metadata.json'))
    metadatas = []
    for i, f in enumerate(fs):
        with open(f, 'r') as fin:
            metadata = json.load(fin)
        metadatas.append(metadata)

    framework.fold_meta = metadatas

    # get the names of any metric columns in the results DataFrame below (same for all folds)
    metric_cols = metadatas[0]['metrics']

    #-----------------------------------------------------
    ## results.csv (one file for all folds)

    # should only be one results.csv file (has all folds)
    # usually results are there even if there was a failure

    logger.debug('Searching for results files with glob: %s', outdir + '/*/scores/results.csv')
    fs = glob.glob(outdir + '/*/scores/results.csv')
    assert len(fs) == 1, len(fs)

    results = pd.read_csv(fs[0])

    gcp_model_paths = []
    if 'model_path' in results.columns:
        # has a saved model for each fold
        for _, row in results.iterrows():
            fold = row['fold']
            model_path = row['model_path']
            if model_path[-1] == '/':
                model_path = model_path[:-1]
            gcp_path = f"tabular-frameworks/{user.id}/{framework.id}/models/{fold}.zip"
            zip_and_upload(model_path, gcp_path)
            gcp_model_paths.append(gcp_path)
            remove_files(model_path)
    else:
        logger.debug('Not uploading any models')
    framework.gcp_model_paths = gcp_model_paths

    base_cols = ['framework', 'version', 'fold', 'type', 'result', 'metric',
                 'duration', 'training_duration', 'predict_duration',
                 'models_count', 'seed', 'info']

    results = results[base_cols + metric_cols]
    results = results.where(pd.notnull(results), None)

    other_metrics = {}
    for metric_name in metric_cols:
        metric_mean = results[metric_name].mean()
        if np.isnan(metric_mean):
            metric_mean = None
        else:
            metric_mean = float(metric_mean)
        other_metrics[metric_name] = metric_mean

    metric_value = results['result'].mean()
    if np.isnan(metric_value):
        metric_value = None
    else:
        metric_value = float(metric_value)

    framework.other_metrics = other_metrics
    framework.fold_results = results.to_dict('records')
    framework.version = results['version'].values[0]
    framework.metric_name = results['metric'].values[0]
    framework.metric_value = metric_value
    framework.problem_type = results['type'].values[0]
    for c in ('duration', 'training_duration', 'predict_duration'):
        setattr(framework, c, float(results[c].sum()))
    framework.models_count = int(results['models_count'].sum())
    framework.info = '\n'.join([str(x) for x in results['info'].values])

    #-----------------------------------------------------
    # predictions.csv (one file per fold)

    # TODO: will these be ordered for n_folds >= 10?
    # - first wildcard is like `randomforest.benchmark.constraint.local.20210611T190804`
    # - second wildcard is over folds (0, 1, ...)
    fs = sorted(glob.glob(outdir + '/*/predictions/files/*/predictions.csv'))

    if len(fs) == 0:
        framework.status = 'FAIL'
        db.session.add(framework)
        db.session.commit()
        remove_files(outdir)
        return

    # should be same length as `preds`
    test_ids = json.loads(framework.test_ids)

    sc = StorageClient()

    for model_fold, local_path in enumerate(fs):
        dset = TabularDataset.query.filter_by(id=test_ids[model_fold]).one()

        # add predictions to DB
        framework_preds = TabularFrameworkPredictions(framework.pk, dset.pk, model_fold, local_path)

        # upload the predictions
        _, ext = os.path.splitext(local_path)
        gcp_path = f"tabular-framework-preds/{framework_preds.id}/predictions{ext}"
        sc.upload(local_path, gcp_path)

        framework_preds.gcp_path = gcp_path
        framework_preds.status = 'SUCCESS'
        db.session.add(framework_preds)

    #-----------------------------------------------------
    # model related (leaderboard.csv, models.txt)

    if 'h2o' in framework.framework_name.lower():
        fs = glob.glob(outdir + '/*/models/files/*/leaderboard.csv')
    else:
        # AutoGluon case
        fs = glob.glob(outdir + '/*/leaderboard/files/*/leaderboard.csv')

    leaderboard = [pd.read_csv(f) for f in fs]
    leaderboard = [df.where(pd.notnull(df), None) for df in leaderboard]
    leaderboard = [df.to_dict('records') for df in leaderboard]

    fs = glob.glob(outdir + '/*/models/files/*/models.txt')
    models_txt = []
    for f in fs:
        with open(f, 'r') as fin:
            models_txt.append(fin.readlines())

    framework.status = 'SUCCESS'
    framework.fold_leaderboard = leaderboard
    framework.fold_model_txt = models_txt

    db.session.add(framework)
    db.session.commit()
    remove_files(outdir)


@shared_task(name='tasks.runbenchmark_on_failure')
def runbenchmark_on_failure(err_request, err_message, err_traceback, framework_pk, outdir):
    """
    Called when `runbenchmark` fails.
    first three args are required by celery
    """

    # get database entries for the run and user
    framework = TabularFramework.query.filter_by(pk=framework_pk).one()
    user = User.query.filter_by(pk=framework.user_pk).one()

    gcp_path = f"tabular-frameworks/{user.id}/{framework.id}.zip"
    zip_and_upload(outdir, gcp_path)

    remove_files(outdir)

    framework.gcp_path = gcp_path
    framework.status = 'FAIL'
    db.session.add(framework)
    db.session.commit()


@shared_task(name='tasks.runbenchmark_predict', bind=True)
def runbenchmark_predict(self, framework_pk, framework_name, dataset_id, constraint_id, outdir, user_dir):
    """
    Calls the `runbenchmark.py` script. Use bind=True so that `self.request` is available.
    """
    logger.info('self.request: %s', str(vars(self.request)))

    # ## update things that should be set before running
    # framework = TabularFramework.query.filter_by(pk=framework_pk).one()
    # framework.outdir = outdir
    # # get and save the celery task id
    # framework.task_id = self.request.id
    # db.session.add(framework)
    # db.session.commit()

    runbenchmark_cmd = get_runbenchmark_cmd()
    cmd = f"{runbenchmark_cmd} {framework_name} {dataset_id} {constraint_id} -o {outdir} -u {user_dir} --logging debug"
    exit_code = run_cmd(cmd)
    if exit_code != 0:
        raise Exception(f"Command exited with non-zero exit status: {exit_code}")
    return exit_code


@shared_task(name='tasks.runbenchmark_predict_on_success')
def runbenchmark_predict_on_success(prev_result, outdir, preds_pk):

    preds = TabularFrameworkPredictions.query.filter_by(pk=preds_pk).one()

    # TODO: will these be ordered for n_folds >= 10?
    # - first wildcard is like `randomforest.benchmark.constraint.local.20210611T190804`
    # - second wildcard is over folds (0, 1, ...)
    fs = sorted(glob.glob(outdir + '/*/predictions/files/*/predictions.csv'))

    if len(fs) == 0:
        preds.status = 'FAIL'
        db.session.add(preds)
        db.session.commit()
        remove_files(outdir)
        return

    # only one fold supported
    local_path = fs[0]

    # upload the predictions
    _, ext = os.path.splitext(local_path)
    gcp_path = f"tabular-framework-preds/{preds.id}/predictions{ext}"
    sc = StorageClient()
    sc.upload(local_path, gcp_path)

    # add predictions to DB
    preds.path = local_path
    preds.gcp_path = gcp_path
    preds.status = 'SUCCESS'
    db.session.add(preds)
    db.session.commit()


@shared_task(name='tasks.runbenchmark_predict_on_failure')
def runbenchmark_predict_on_failure(err_request, err_message, err_traceback, preds_pk):
    """
    first three args are required by celery
    """
    preds = TabularFrameworkPredictions.query.filter_by(pk=preds_pk).one()
    preds.status = 'FAIL'
    db.session.add(preds)
    db.session.commit()
