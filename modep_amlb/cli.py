"""
"""
import sys
import logging
import fire

from modep_amlb import app
from modep_amlb.v1.api_def import setup_train, setup_predict
from modep_amlb.tasks import (
    get_runbenchmark_cmd,
    run_cmd,
    on_success_train,
    on_failure_train,
    on_success_predict,
    on_failure_predict,
)

logger = logging.getLogger(__name__)


def to_list(x):
    if isinstance(x, str):
        return x.split(',')
    elif isinstance(x, list):
        return x
    else:
        raise Exception(f'Unknown type: {type(x)}')


def run_benchmark(framework_name, outdir):
    python_cmd = get_runbenchmark_cmd()
    exit_code = run_cmd(f"{python_cmd} {framework_name} benchmark constraint -o {outdir} -u {outdir} --logging debug")
    return exit_code


def train(
        framework_pk: int,
        framework_name: str,
        train_ids: str,
        test_ids: str,
        target: str,
        max_runtime_seconds: int,
        cores: int,
        outdir=None,
):
    logger.info('locals')
    logger.info(locals())

    train_ids = to_list(train_ids)
    test_ids = to_list(test_ids)

    with app.app_context():
        try:
            outdir = setup_train(outdir, train_ids, test_ids, target, max_runtime_seconds, cores)
        except Exception as e:
            logger.exception(e)
            # no outdir to upload
            on_failure_train(framework_pk, info='failure in training setup')
            sys.exit(-1)
        try:
            exit_code = run_benchmark(framework_name, outdir)
            if exit_code != 0:
                on_failure_train(framework_pk, outdir)
                sys.exit(exit_code)
            on_success_train(framework_pk, outdir)
            sys.exit(0)
        except Exception as e:
            logger.exception(e)
            on_failure_train(framework_pk, outdir)
            sys.exit(-1)


def predict(
        preds_pk: int,
        gcp_model_path: int,
        framework_name: str,
        train_ids: str,
        test_ids: str,
        target: str,
        max_runtime_seconds: int,
        cores: int,
        outdir=None,
):
    logger.info('locals')
    logger.info(locals())

    train_ids = to_list(train_ids)
    test_ids = to_list(test_ids)

    with app.app_context():
        try:
            outdir = setup_predict(gcp_model_path, outdir, train_ids, test_ids, target, max_runtime_seconds, cores)
        except Exception as e:
            logger.exception(e)
            on_failure_predict(preds_pk, info='failure in prediction setup')
            sys.exit(-1)
        try:
            exit_code = run_benchmark(framework_name, outdir)
            if exit_code != 0:
                on_failure_predict(preds_pk, outdir)
                sys.exit(exit_code)
            on_success_predict(preds_pk, outdir)
            sys.exit(0)
        except Exception as e:
            logger.exception(e)
            on_failure_predict(preds_pk, outdir)
            sys.exit(-1)

if __name__ == '__main__':
    fire.Fire()
