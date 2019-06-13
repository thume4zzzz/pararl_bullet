import traceback
import tempfile
import six
from path import Path
from abc import ABCMeta, abstractmethod

import tensorflow as tf

from ..util import json, ec2, tee_out

gfile = tf.gfile


@six.add_metaclass(ABCMeta)
class Experiment(object):

    def __init__(self, experiment_name, out_dir='out/'):
        self.experiment_name = experiment_name
        self.out_dir = Path(out_dir)

    @abstractmethod
    def initialize(self, out_dir):
        pass

    @abstractmethod
    def to_dict(self):
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, params):
        pass

    def run(self, remote=False, **kwargs):
        out_dir = self.out_dir / self.experiment_name
        if remote is False:  # local
            if not gfile.Exists(out_dir):
                gfile.MakeDirs(out_dir)
            with gfile.GFile(out_dir / "params.json", 'w') as fp:
                json.dump(self, fp)
            try:
                with tee_out(out_dir):
                    self.initialize(out_dir)
                    self.run_experiment(out_dir)
            except:
                traceback.print_exc()
                with gfile.GFile(out_dir / 'exception.log', 'w') as fp:
                    traceback.print_exc(file=fp)
        else:
            assert out_dir[:5] == "s3://", "Must be dumping to s3"
            with gfile.GFile(out_dir / "params.json", 'w') as fp:
                fp.write(json.dumps(self))
            while True:
                try:
                    with gfile.GFile(out_dir / "params.json", 'r') as fp:
                        json.load(fp)
                    break
                except:
                    print("Failed experiment upload...trying again")
                    with gfile.GFile(out_dir / "params.json", 'w') as fp:
                        fp.write(json.dumps(self))
            return ec2.run_remote(out_dir / "params.json", **kwargs)
        return self

    @abstractmethod
    def run_experiment(self, out_dir):
        pass

    def __getstate__(self):
        return self.to_dict()

    def __setstate__(self, state):
        self.__dict__.update(self.from_dict(state).__dict__)
