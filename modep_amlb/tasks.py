import os
import json
import glob
import shutil
import logging
import subprocess
import numpy as np
import pandas as pd

from celery import shared_task

from app_utils.models import (
    db,
    User,
    TabularFramework,
    TabularFrameworkPredictions,
)
from app_utils.io import StorageClient

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
    if os.path.exists('/bench/venv/bin'):
        # running in docker container
        return False
    else:
        return True

def python_cmd():
    # run python command
    if is_local():
        # local conda
        return 'python -W ignore /home/jimmie/git/mlapi/automlbenchmark/runbenchmark.py'
    else:
        # docker
        return '/bench/venv/bin/python3 -W ignore /bench/automlbenchmark/runbenchmark.py'

def remove_files(outdir):
    if is_local():
        return

    if os.path.exists(outdir):
        logger.info("Deleting outdir to save space: %s", outdir)
        shutil.rmtree(outdir)

    zip_file = outdir + '.zip'
    if os.path.exists(zip_file):
        logger.info("Deleting outdir zip to save space: %s", zip_file)
        os.remove(zip_file)

def zip_and_upload(outdir, gcp_path):
    # logger.info('SKIPPING ZIP UPLOAD')
    # return

    # exclude input data
    data_dir = os.path.join(outdir, 'input-data')
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)

    # create zip file of output
    shutil.make_archive(outdir, 'zip', outdir)

    # upload
    sc = StorageClient()
    sc.upload(outdir + '.zip', gcp_path)

@shared_task(name='tasks.run_benchmark', bind=True)
def run_benchmark(self, framework_pk, framework_name, dataset_id, constraint_id, outdir, user_dir):
    logger.info('self.request: %s', str(vars(self.request)))

    ## update things that should be set before running
    framework = TabularFramework.query.filter_by(pk=framework_pk).one()
    framework.outdir = outdir
    framework.task_id = self.request.id
    db.session.add(framework)
    db.session.commit()

    py = python_cmd()
    cmd = f"{py} {framework_name} {dataset_id} {constraint_id} -o {outdir} -u {user_dir} --logging debug"
    exit_code = run_cmd(cmd)
    if exit_code != 0:
        raise Exception('Command exited with non-zero exit status: %i' % exit_code)
    return exit_code

@shared_task(name='tasks.on_success')
def on_success(prev_result, framework_pk, outdir):
    """ prev_result is not used (required to be passed b/c of the task linkage with `run_benchmark`) """

    framework = TabularFramework.query.filter_by(pk=framework_pk).one()
    user = User.query.filter_by(pk=framework.user_pk).one()

    gcp_path = f"tabular-frameworks/{user.id}/{framework.id}.zip"

    zip_and_upload(outdir, gcp_path)

    framework.gcp_path = gcp_path

    #-----------------------------------------------------
    # metadata.json (one file per fold)

    fs = sorted(glob.glob(outdir + '/*/predictions/files/*/metadata.json'))

    metadatas = []
    for i, f in enumerate(fs):
        metadata = json.load(open(f, 'r'))

        for k in ['input_dir', 'output_dir', 'output_predictions_file', 'output_metadata_file']:
            # delete keys that we don't want to expose to user
            del metadata[k]

        metadatas.append(metadata)

    framework.fold_meta = metadatas
    db.session.add(framework)
    db.session.commit()

    #-----------------------------------------------------
    ## results.csv (one file for all folds)

    # usually results are there even if there was a failure
    fs = glob.glob(outdir + '/*/scores/results.csv')
    results = pd.read_csv(fs[0])

    base_cols = ['framework', 'version', 'fold', 'type', 'result', 'metric',
                 'duration', 'training_duration', 'predict_duration',
                 'models_count', 'seed', 'info']

    metric_cols = metadatas[0]['metrics']
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

    # first wildcard is something like `randomforest.benchmark.constraint.local.20210611T190804`
    # second wildcard is over folds (0, 1, ...)
    fs = sorted(glob.glob(outdir + '/*/predictions/files/*/predictions.csv'))

    if len(fs) == 0:
        framework.status = 'FAIL'
        db.session.add(framework)
        db.session.commit()
        remove_files(outdir)
        return

    preds = []
    for i, f in enumerate(fs):
        preds.append(pd.read_csv(f).to_dict('records'))

    # framework.fold_predictions = preds        
    framework_preds = TabularFrameworkPredictions(framework.pk, preds)
    db.session.add(framework_preds)

    #-----------------------------------------------------
    # model related (leaderboard.csv, models.txt)

    if 'h2o' in framework.framework_id.lower():
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

@shared_task(name='tasks.on_failure')
def on_failure(err_request, err_message, err_traceback, framework_pk, outdir):
    """ first three args are required by celery """

    framework = TabularFramework.query.filter_by(pk=framework_pk).one()
    user = User.query.filter_by(pk=framework.user_pk).one()

    gcp_path = f"tabular-frameworks/{user.id}/{framework.id}.zip"
    zip_and_upload(outdir, gcp_path)

    remove_files(outdir)

    framework.gcp_path = gcp_path
    framework.status = 'FAIL'
    db.session.add(framework)
    db.session.commit()
