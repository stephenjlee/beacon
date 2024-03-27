import sys, os, json

# os.environ["OMP_NUM_THREADS"] = "1"
from datetime import datetime

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
sys.path.append(os.environ.get("LDF_ROOT"))

from ldf.models.model_ldf import LdfModel
import ldf.models.nn_define_models as dm

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

load_dotenv(find_dotenv())

from ldf.utils.utils_bebi import \
    plot_posterior_predictive_checks, \
    plot_posterior_predictive_and_metrics, \
    plot_posterior_samples

from ldf.visualization.plotting_functions import plot_prec_recall_roc

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
sys.path.append(os.environ.get("PROJECT_ROOT"))

import matplotlib.pyplot as plt


class LdfBetaBinomModel(LdfModel):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        default_kwargs = {
            'plotpostpred': True,
            'plotpost': True
        }

        merged_kwargs = {}
        for key in default_kwargs.keys():
            if key in kwargs.keys():
                merged_kwargs[key] = kwargs[key]
            else:
                merged_kwargs[key] = default_kwargs[key]

        self.plotpostpred = merged_kwargs['plotpostpred']
        self.plotpost = merged_kwargs['plotpost']

    def fit(self,
            X=None,
            y=None,
            n=None,
            X_val=None,
            y_val=None,
            n_val=None,
            load_model=False,
            save_model=True):

        if type(self.params) is not dict:
            self.params = json.loads(self.params)

        self.define_distn()

        self.X_ = X
        self.y_ = y
        self.load_model_ = load_model

        # setup
        if self.gpu == 0:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

        # format output
        out_basename = self.model_name \
                       + f'_{self.error_metric}' \
                       + f'_{datetime.now().strftime("%Y%m%d%H%M%S")}' \
                       + f'_t{self.fold_num_test}' \
                       + f'_v{self.fold_num_val}' \
                       + f'_do{round(self.do_val * 100)}' \
                       + f'_rm{self.reg_mode}' \
                       + f'_rv{round(self.reg_val * 100)}'

        # make output directory
        self.output_dir = os.path.join(self.output_dir, out_basename)
        print(f'output dir: {self.output_dir}')
        os.makedirs(self.output_dir, exist_ok=True)

        args = {key: val for key, val in self.__dict__.items() if
                isinstance(val, int) or isinstance(val, str) or isinstance(val, dict) or isinstance(val,
                                                                                                    float) or isinstance(
                    val, bool)}
        # save params to external csv and json files
        with open(os.path.join(self.output_dir, 'args.json'), "w") as outfile:
            json.dump(args, outfile, indent=2)

        # load model
        if load_model:

            self.model_ = dm.load_model(self.load_json,
                                        self.load_weights,
                                        self.distn)

            self.history_ = None
            self.train_mean_nll_ = None
            self.train_nlls_ = None

        elif (X is None):

            raise Exception(f"X is None, but load_model is False. We cannot train!")

        elif (y is None):

            raise Exception(f"y is None, but load_model is False. We cannot train!")

        else:

            output_dim = 2
            # incorporate hyperparams into y_train
            yn_train = np.stack((y, n), axis=1)

            self.model_, _ = \
                dm.define_model(input_dim=X.shape[1],
                                output_dim=output_dim,
                                distn=self.distn,
                                do_val=self.do_val,
                                learning_rate_training=self.learning_rate_training,
                                model_name=self.model_name,
                                reg_mode=self.reg_mode,
                                reg_val=self.reg_val)

            # get prob_callback
            callbacks_prob = \
                dm.get_callbacks_prob(
                    red_lr_on_plateau=self.red_lr_on_plateau,
                    red_factor=self.red_factor,
                    red_patience=self.red_patience,
                    red_min_lr=self.red_min_lr,
                    verbose=self.verbose,
                    es_patience=self.es_patience)

            if X_val is not None:
                yn_val = np.stack((y_val, n_val), axis=1)
                self.history_ = self.model_.fit(X, yn_train,
                                                epochs=self.train_epochs,
                                                batch_size=self.batch_size,
                                                verbose=self.verbose,
                                                workers=1,
                                                validation_data=(X_val, yn_val),
                                                callbacks=callbacks_prob,
                                                use_multiprocessing=False,
                                                shuffle=True
                                                )
            else:
                self.history_ = self.model_.fit(X, yn_train,
                                                epochs=self.train_epochs,
                                                batch_size=self.batch_size,
                                                verbose=self.verbose,
                                                workers=1,
                                                validation_split=0.1,
                                                callbacks=callbacks_prob,
                                                use_multiprocessing=False,
                                                shuffle=True
                                                )

        if save_model:
            # save model definition as .json and model weights as .h5
            model_json = self.model_.to_json()  # serialize model to JSON
            # with open(f'{output_dir}/model_{fold}.json', 'w') as json_file:
            with open('{}/model.json'.format(self.output_dir), 'w') as json_file:
                json_file.write(model_json)
            # model.save_weights(f'{output_dir}/model_{fold}.h5')  # serialize weights to HDF5
            self.model_.save_weights('{}/model.h5'.format(self.output_dir))  # serialize weights to HDF5
            print('Saved model to disk')

        return self

    def predict(self, X=None, n=None):

        yhat = self.model_.predict(X.astype(np.float64), verbose=2)

        mean_preds, \
            preds_params, \
            preds_params_flat = \
            self.distn.interpret_predict_output_ts(yhat, n)

        return mean_preds, preds_params, preds_params_flat

    def predict_for_train(self,
                          train_x,
                          train_y,
                          n_train):

        self.train_mean_preds, \
            self.train_preds_params, \
            self.train_preds_params_flat = \
            self.predict(X=train_x, n=n_train)

        self.train_mean_nll, \
            self.train_nlls = \
            self.distn.compute_nll(self.train_preds_params,
                                   train_y,
                                   n_train)
        self.train_acc = accuracy_score(train_y, (self.train_mean_preds >= 0.5))

        return self

    def predict_for_val(self, val_x, val_y, n_val):

        self.val_mean_preds, \
            self.val_preds_params, \
            self.val_preds_params_flat = \
            self.predict(X=val_x, n=n_val)

        self.val_mean_nll, \
            self.val_nlls = \
            self.distn.compute_nll(self.val_preds_params,
                                   val_y,
                                   n_val)

        self.val_acc = accuracy_score(val_y, (self.val_mean_preds >= 0.5))

        return self

    def predict_for_test(self, test_x, test_y, n_test):

        self.test_mean_preds, \
            self.test_preds_params, \
            self.test_preds_params_flat = \
            self.predict(X=test_x, n=n_test)

        self.test_mean_nll, \
            self.test_nlls = \
            self.distn.compute_nll(self.test_preds_params,
                                   test_y,
                                   n_test)

        self.test_acc = accuracy_score(test_y, (self.test_mean_preds >= 0.5))

        # self.test_cdfs = self.distn.cdf_params(test_y, preds_params)

        # # evaluate model calibration
        # utf.eval_ts_prob_calibration(self.output_dir,
        #                              self.test_cdfs,
        #                              self.fold_num_test)

        return self

    def save_summary(self,
                     x_train,
                     x_val,
                     x_test,
                     y_train,
                     y_val,
                     y_test,
                     n_train,
                     n_val,
                     n_test,
                     ids_train,
                     ids_val,
                     ids_test):

        self.metrics_df = pd.DataFrame(columns=['fold_num', 'metric', 'val'])
        # 
        # inf_indices = np.where(np.isinf(self.test_nlls))
        # neginf_indices = np.where(np.isneginf(self.test_nlls))
        # nan_indices = np.where(np.isnan(self.test_nlls))

        # metrics to log for each fold
        metrics_to_log = [
            ('x_train', x_train),
            ('x_val', x_val),
            ('x_test', x_test),
            ('y_train', y_train),
            ('y_val', y_val),
            ('y_test', y_test),
            ('n_train', n_train),
            ('n_val', n_val),
            ('n_test', n_test),
            ('ids_train', ids_train),
            ('ids_val', ids_val),
            ('ids_test', ids_test),
            ('train_preds_params', self.train_preds_params),
            ('val_preds_params', self.val_preds_params),
            ('test_preds_params', self.test_preds_params),
            ('train_mean_preds', self.train_mean_preds),
            ('val_mean_preds', self.val_mean_preds),
            ('test_mean_preds', self.test_mean_preds),
            ('train_acc', self.train_acc),
            ('val_acc', self.val_acc),
            ('test_acc', self.test_acc),
            ('train_mean_nll', self.train_mean_nll),
            ('val_mean_nll', self.val_mean_nll),
            ('test_mean_nll', self.test_mean_nll),
            ('train_nlls', self.train_nlls),
            ('val_nlls', self.val_nlls),
            ('test_nlls', self.test_nlls),
            ('history_loss', self.history_.history['loss']),
            ('history_val_loss', self.history_.history['val_loss'])
        ]
        for run_metric_name, run_metric_val in metrics_to_log:
            metric_dict = {
                'subset': 'crossval',
                'metric': run_metric_name,
                'val': run_metric_val
            }
            self.metrics_df = pd.concat([self.metrics_df, pd.DataFrame([metric_dict])], ignore_index=True)

        self.metrics_df.to_pickle(os.path.join(self.output_dir, 'metrics_df.p'))
        self.metrics_df.to_csv(os.path.join(self.output_dir, 'metrics_df.csv'))

        return self

    def return_metrics_df(self):

        return self.metrics_df

    def plot_postpred(self,
                      ids_test,
                      y_train,
                      y_test,
                      n_test):

        pred_incr = plot_posterior_predictive_checks(ids_test,
                                                     0,
                                                     y_test,
                                                     n_test,
                                                     self.params,
                                                     self.test_preds_params,
                                                     gamma=0.9,
                                                     plot=True,
                                                     output_dir=self.output_dir)

        y_pred_train = (self.train_mean_preds > 0.5).astype(int)
        y_pred_test = (self.test_mean_preds > 0.5).astype(int)

        # # plot for train
        prec_recall_roc_output_path_train = os.path.join(self.output_dir, f'prec_recall_roc_train.pdf')
        plot_prec_recall_roc(y_train, y_pred_train, 1, prec_recall_roc_output_path_train)
        # # plot for test
        prec_recall_roc_output_path_test = os.path.join(self.output_dir, f'prec_recall_roc_test.pdf')
        plot_prec_recall_roc(y_test, y_pred_test, 1, prec_recall_roc_output_path_test)

        plot_posterior_predictive_and_metrics(ids_test,
                                              0,
                                              y_test,
                                              self.params,
                                              self.test_preds_params,
                                              output_dir=self.output_dir)

    def plot_posterior(self,
                       y_test,
                       n_test):

        plot_posterior_samples(y_test,
                               n_test,
                               self.params,
                               self.test_preds_params,
                               output_dir=self.output_dir)

    def plot_history(self):

        plt.figure(figsize=(16, 8))
        plt.plot(self.history_.history['loss'])
        plt.plot(self.history_.history['val_loss'])
        os.makedirs(self.output_dir, exist_ok=True)
        plt.savefig(os.path.join(self.output_dir, f'history_{0}.pdf'))
        plt.close()

    def plot_calibration(self,
                         y_train,
                         y_val,
                         y_test):

        y_list = [y_train, y_val, y_test]
        preds_list = [self.train_mean_preds, self.val_mean_preds, self.test_mean_preds]
        str_list = ['train', 'val', 'test']

        for y, preds, str in zip(y_list, preds_list, str_list):

            y_mean = np.around(preds, decimals=2)

            calibration_df = pd.DataFrame.from_dict({
                'y_test': y,
                'y_mean_test': y_mean,
            })

            decs = np.sort(calibration_df['y_mean_test'].unique())
            empirs = []
            counts = []
            for dec in decs:
                calibration_df[calibration_df['y_mean_test'] == dec].mean()
                empir = calibration_df[calibration_df['y_mean_test'] == dec].mean()['y_test']
                empirs.append(empir)
                counts.append(calibration_df[calibration_df['y_mean_test'] == dec].shape[0])

            decs = np.array(decs) * 100.
            empirs = np.array(empirs) * 100
            counts = np.array(counts)

            fig, axs = plt.subplots(2, figsize=(8, 8))
            # fig.suptitle('Model calibration plots')
            axs[0].scatter(decs, empirs)
            axs[0].set_title('Empirical vs predicted electricity access rate')
            axs[0].set(xlabel='$E[y|x]$, expected value of posterior-predictive distribution (%)',
                       ylabel='Empirical access rate (%)')
            axs[1].scatter(decs, counts)
            axs[1].set_title('Number of samples vs predicted electricity access rate')
            axs[1].set(xlabel='$E[y|x]$, expected value of posterior-predictive distribution (%)',
                       ylabel='Number of samples')
            plt.tight_layout()
            output_path = os.path.join(self.output_dir, f'calibration_{str}.pdf')
            plt.savefig(output_path)
            plt.close('all')

            print(f'saved model calibration plots')
