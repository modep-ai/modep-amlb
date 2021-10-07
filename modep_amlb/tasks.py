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
from modep_common.enums import JobStatus

logger = logging.getLogger(__name__)

MODEP_FILE_CLEANUP = os.environ.get('MODEP_FILE_CLEANUP', False)


def run_cmd(cmd):
    """ Run a shell command and print the output as it runs """
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


def get_runbenchmark_cmd():
    """ Returns the path to the python command """
    if os.path.exists('/bench/venv/bin'):
        # when running in docker container
        return '/bench/venv/bin/python3 -W ignore /bench/automlbenchmark/runbenchmark.py'
    else:
        # when running locally
        return 'python -W ignore /home/jimmie/git/mlapi/automlbenchmark/runbenchmark.py'


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
    if os.path.exists(outdir):
        logger.info("Deleting outdir to save space: %s", outdir)
        shutil.rmtree(outdir)

    zip_file = outdir + '.zip'
    if os.path.exists(zip_file):
        logger.info("Deleting outdir zip to save space: %s", zip_file)
        os.remove(zip_file)


def on_success_train(framework_pk, outdir):
    framework = TabularFramework.query.filter_by(pk=framework_pk).one()

    try:
        # upload entire output folder, removes input-data subdir with downloaded models in it
        gcp_path = f"tabular-frameworks/{framework.id}/outdir.zip"
        zip_and_upload(outdir, gcp_path)
        framework.gcp_path = gcp_path
        # commit now so that we can fetch the logs for any exceptions
        db.session.add(framework)
        db.session.commit()

        #-----------------------------------------------------
        # metadata.json (one file per fold)

        fs = sorted(glob.glob(outdir + '/*/predictions/modep/*/metadata.json'))
        metadatas = []
        for i, f in enumerate(fs):
            with open(f, 'r') as fin:
                metadata = json.load(fin)
            metadatas.append(metadata)

        framework.fold_meta = metadatas
        db.session.add(framework)
        db.session.commit()

        if len(metadatas) == 0:
            framework.status = JobStatus.FAIL.name
            framework.info = 'no model metadata'
            db.session.add(framework)
            db.session.commit()
            if MODEP_FILE_CLEANUP:
                remove_files(outdir)
            return JobStatus.FAIL.name

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
                if not isinstance(model_path, str):
                    # there was an error saving the model
                    gcp_model_paths.append(None)
                    continue
                if model_path[-1] == '/':
                    model_path = model_path[:-1]
                gcp_path = f"tabular-frameworks/{framework.id}/models/{fold}.zip"
                zip_and_upload(model_path, gcp_path)
                gcp_model_paths.append(gcp_path)
                if MODEP_FILE_CLEANUP:
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
        framework.version = str(results['version'].values[0])
        framework.metric_name = results['metric'].values[0]
        framework.metric_value = metric_value
        framework.problem_type = results['type'].values[0]
        for c in ('duration', 'training_duration', 'predict_duration'):
            setattr(framework, c, float(results[c].sum()))
        framework.models_count = int(results['models_count'].sum())
        framework.info = '\n'.join([str(x) for x in results['info'].values])
        db.session.add(framework)
        db.session.commit()

        #-----------------------------------------------------
        # predictions.csv (one file per fold)

        # TODO: will these be ordered for n_folds >= 10?
        # - first wildcard is like `randomforest.benchmark.constraint.local.20210611T190804`
        # - second wildcard is over folds (0, 1, ...)
        fs = sorted(glob.glob(outdir + '/*/predictions/modep/*/predictions.csv'))

        if len(fs) == 0:
            framework.status = JobStatus.FAIL.name
            framework.info += ', no predictions'
            db.session.add(framework)
            db.session.commit()
            if MODEP_FILE_CLEANUP:
                remove_files(outdir)
            return JobStatus.FAIL.name

        # should be same length as `preds`
        test_ids = json.loads(framework.test_ids)

        sc = StorageClient()

        for model_fold, local_path in enumerate(fs):
            dset = TabularDataset.query.filter_by(id=test_ids[model_fold]).one()

            # add predictions to DB
            framework_preds = TabularFrameworkPredictions(framework.user_pk, framework.pk, dset.pk, model_fold, local_path)

            # upload the predictions
            _, ext = os.path.splitext(local_path)
            gcp_path = f"tabular-framework-predictions/{framework_preds.id}/predictions{ext}"
            sc.upload(local_path, gcp_path)

            framework_preds.gcp_path = gcp_path
            framework_preds.status = JobStatus.SUCCESS.name
            db.session.add(framework_preds)

        db.session.commit()

        #-----------------------------------------------------
        # model related (leaderboard.csv, models.txt)

        if 'h2o' in framework.framework_name.lower():
            fs = glob.glob(outdir + '/*/models/modep/*/leaderboard.csv')
        else:
            # AutoGluon case
            fs = glob.glob(outdir + '/*/leaderboard/modep/*/leaderboard.csv')

        leaderboard = [pd.read_csv(f) for f in fs]
        leaderboard = [df.where(pd.notnull(df), None) for df in leaderboard]
        leaderboard = [df.to_dict('records') for df in leaderboard]

        fs = glob.glob(outdir + '/*/models/modep/*/models.txt')
        models_txt = []
        for f in fs:
            with open(f, 'r') as fin:
                models_txt.append(fin.readlines())

        framework.status = JobStatus.SUCCESS.name
        framework.fold_leaderboard = leaderboard
        framework.fold_model_txt = models_txt

        db.session.add(framework)
        db.session.commit()

        if MODEP_FILE_CLEANUP:
            remove_files(outdir)

    except Exception as e:
        framework.status = JobStatus.FAIL.name
        framework.info += ', error processing results'
        db.session.add(framework)
        db.session.commit()
        if MODEP_FILE_CLEANUP:
            remove_files(outdir)
        logger.exception(e)
        raise


def on_failure_train(framework_pk, outdir=None, info=None):
    # get database entries for the run and user
    framework = TabularFramework.query.filter_by(pk=framework_pk).one()

    if outdir is not None:
        # upload entire output folder
        framework.gcp_path = f"tabular-frameworks/{framework.id}/outdir.zip"
        zip_and_upload(outdir, framework.gcp_path)
        if MODEP_FILE_CLEANUP:
            remove_files(outdir)

    # mark as failed
    framework.status = JobStatus.FAIL.name
    if info is not None:
        framework.info = info
    else:
        framework.info += ', failure in training'
    db.session.add(framework)
    db.session.commit()


def on_success_predict(preds_pk, outdir):
    preds = TabularFrameworkPredictions.query.filter_by(pk=preds_pk).one()

    fs = sorted(glob.glob(outdir + '/*/predictions/modep/*/predictions.csv'))

    if len(fs) == 0:
        preds.status = JobStatus.FAIL.name
        preds.info = 'No predictions were created by the model'
        db.session.add(preds)
        db.session.commit()
        if MODEP_FILE_CLEANUP:
            remove_files(outdir)
        return JobStatus.FAIL.name
    elif len(fs) > 1:
        preds.status = JobStatus.FAIL.name
        preds.info = 'Multiple sets of predictions were created by the model'
        db.session.add(preds)
        db.session.commit()
        if MODEP_FILE_CLEANUP:
            remove_files(outdir)
        return JobStatus.FAIL.name

    # predict only works with one dataset, so there's only one predictions file
    local_path = fs[0]

    # upload the predictions
    _, ext = os.path.splitext(local_path)
    gcp_path = f"tabular-framework-predictions/{preds.id}/predictions{ext}"
    sc = StorageClient()
    sc.upload(local_path, gcp_path)

    # add predictions to DB
    preds.path = local_path
    preds.gcp_path = gcp_path
    preds.status = JobStatus.SUCCESS.name
    db.session.add(preds)
    db.session.commit()


def on_failure_predict(preds_pk, outdir=None, info=None):
    preds = TabularFrameworkPredictions.query.filter_by(pk=preds_pk).one()
    preds.status = JobStatus.FAIL.name
    if info is not None:
        preds.info = info
    else:
        preds.info = 'failure getting predictions'
    db.session.add(preds)
    db.session.commit()

    
@shared_task(name='tasks.runbenchmark', bind=True)
def runbenchmark(self, framework_pk, framework_name, dataset_id, constraint_id, outdir, user_dir):
    """
    Calls the `runbenchmark.py` script.
    """
    # Add the celery task ID to the DB
    framework = TabularFramework.query.filter_by(pk=framework_pk).one()
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
    on_success_train(framework_pk, outdir)


@shared_task(name='tasks.runbenchmark_on_failure')
def runbenchmark_on_failure(err_request, err_message, err_traceback, framework_pk, outdir):
    """
    Called when `runbenchmark` fails.
    first three args are required by celery
    """
    on_failure_train(framework_pk, outdir)


@shared_task(name='tasks.runbenchmark_predict', bind=True)
def runbenchmark_predict(self, framework_pk, framework_name, dataset_id, constraint_id, outdir, user_dir):
    """
    Calls the `runbenchmark.py` script. Use bind=True so that `self.request` is available.
    """
    runbenchmark_cmd = get_runbenchmark_cmd()
    cmd = f"{runbenchmark_cmd} {framework_name} {dataset_id} {constraint_id} -o {outdir} -u {user_dir} --logging debug"
    exit_code = run_cmd(cmd)
    if exit_code != 0:
        raise Exception(f"Command exited with non-zero exit status: {exit_code}")
    return exit_code


@shared_task(name='tasks.runbenchmark_predict_on_success')
def runbenchmark_predict_on_success(prev_result, preds_pk, outdir):
    on_success_predict(preds_pk, outdir)


@shared_task(name='tasks.runbenchmark_predict_on_failure')
def runbenchmark_predict_on_failure(err_request, err_message, err_traceback, preds_pk, outdir):
    on_failure_predict(preds_pk, outdir)
