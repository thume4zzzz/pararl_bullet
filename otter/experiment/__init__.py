from functools import partial
import multiprocessing

from ..util import json
from .common import Experiment
from .util import sweep


import tensorflow as tf

gfile = tf.gfile

def from_json(fp):
    if isinstance(fp, str):
        with gfile.GFile(fp, 'r') as f:
            params = json.load(f)
    else:
        params = json.load(fp)
    return EXPERIMENT_MAP[params['experiment_type']].from_dict(params)

def expand_params(params):
    params = params.copy()
    for k, v in params.items():
        if isinstance(v, dict):
            v_s = list(expand_params(v))
            if len(v_s) > 1:
                for v_, exps in v_s:
                    params_ = params.copy()
                    params_[k] = v_
                    for new_params, expansions in expand_params(params_):
                        yield new_params, [('%s{%s}' % (k, a), b) for a, b in exps] + expansions
                return
        if isinstance(v, sweep):
            for v_, name in v:
                params_ = params.copy()
                params_[k] = v_
                for new_params, expansions in expand_params(params_):
                    yield new_params, expansions + [(k, name)]
            return
    yield params, []

def run(params, **kwargs):
    num_threads = kwargs.pop('num_threads', 1)
    experiments = []
    for param, expansions in expand_params(params):
        if len(expansions) > 0:
            param['experiment_name'] += '_' + '.'.join(["%s-%s" % (a, b) for a, b in expansions])
        experiments.append(param)
    if len(experiments) > 1:
        print("Found %u experiments!" % len(experiments))
    if num_threads > 1:
        with multiprocessing.Pool(num_threads) as p:
            p.map(partial(run_experiment, **kwargs), experiments)
    else:
        for experiment in experiments:
            run_experiment(experiment, **kwargs)

def run_experiment(params, **kwargs):
#    from .vae import TrainVAE
    #from .solar import Solar
    from .myexp import Myexp
    EXPERIMENTS = [Myexp ]

    EXPERIMENT_MAP = {}
    for experiment in EXPERIMENTS:
        EXPERIMENT_MAP[experiment.experiment_type] = experiment

    experiment = EXPERIMENT_MAP[params['experiment_type']].from_dict(params)
    return experiment.run(**kwargs)
