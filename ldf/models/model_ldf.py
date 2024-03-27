import sys, os, pickle
import numpy as np
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
sys.path.append(os.environ.get("LDF_ROOT"))

os.environ["OMP_NUM_THREADS"] = "1"

from sklearn.base import BaseEstimator


class LdfModel(BaseEstimator):

    def __init__(self, **kwargs):

        default_kwargs = {
            'hash': '555',
            'red_lr_on_plateau': True,
            'error_metric': 'gapo',
            'num_folds': 26,
            'fold_num_test': 1,
            'fold_num_val': 1,
            'load_json': 'model.json',
            'load_weights': 'model.h5',
            'output_dir': os.path.join(os.environ.get("PROJECT_ROOT"), 'out', 'bademo'),
            'train_epochs': 10000,
            'es_patience': 200,
            'n_rand_restarts': 1,
            'verbose': 2,
            'batch_size': 256,
            'params': '{"a0": 0.0001, "b0": 0.0001, "nu": 1.0}',
            'gpu': 0,
            'do_val': 0.01,
            'reg_mode': 'L2',
            'reg_val': 0.00,
            'learning_rate_training': 0.0001,
            'model_name': 'TS_MLP_X3LARGE',
            'red_factor': 0.5,
            'red_patience': 50,
            'red_min_lr': 0.0000000001
        }

        merged_kwargs = {}
        for key in default_kwargs.keys():
            if key in kwargs.keys():
                merged_kwargs[key] = kwargs[key]
            else:
                merged_kwargs[key] = default_kwargs[key]

        self.error_metric = merged_kwargs['error_metric']
        self.num_folds = merged_kwargs['num_folds']
        self.fold_num_test = merged_kwargs['fold_num_test']
        self.fold_num_val = merged_kwargs['fold_num_val']
        self.load_json = merged_kwargs['load_json']
        self.load_weights = merged_kwargs['load_weights']
        self.output_dir = merged_kwargs['output_dir']
        self.train_epochs = merged_kwargs['train_epochs']
        self.es_patience = merged_kwargs['es_patience']
        self.n_rand_restarts = merged_kwargs['n_rand_restarts']
        self.verbose = merged_kwargs['verbose']
        self.batch_size = merged_kwargs['batch_size']
        self.params = merged_kwargs['params']
        self.gpu = merged_kwargs['gpu']
        self.do_val = merged_kwargs['do_val']
        self.reg_mode = merged_kwargs['reg_mode']
        self.reg_val = merged_kwargs['reg_val']
        self.learning_rate_training = merged_kwargs['learning_rate_training']
        self.model_name = merged_kwargs['model_name']
        self.red_lr_on_plateau = merged_kwargs['red_lr_on_plateau']
        self.red_factor = merged_kwargs['red_factor']
        self.red_patience = merged_kwargs['red_patience']
        self.red_min_lr = merged_kwargs['red_min_lr']
        self.hash = merged_kwargs['hash']

        self.X_ = None
        self.y_ = None
        self.model_ = None

    def define_distn(self):

        if self.error_metric == 'betabinom':
            from ldf.models.distn_betabinom import DistnBetaBinom
            self.distn = DistnBetaBinom(self.params)
        return

    def fit(self, X=None, y=None, load_model=False):
        pass

    def predict(self, X=None):

        yhat = self.model_.predict(X.astype(np.float64), verbose=0)

        mean_preds, \
            preds_params, \
            preds_params_flat = \
            self.distn.interpret_predict_output_ts(yhat)

        return mean_preds, preds_params, preds_params_flat

    def save_pickle(self):

        pickle.dump(self, open(os.path.join(self.output_dir, 'model.p'), 'wb'))
        print('saved pickle')

    def get_output_dir(self):
        return self.output_dir
