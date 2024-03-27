import sys, os, json, shutil, ast, hashlib, pickle, cv2
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
sys.path.append(os.environ.get("PROJECT_ROOT"))

from ldf.models.model_ldf_scalar import LdfScalarModel
import ldf.models.nn_define_models as dm
import ldf.utils.utils_stat as us
import ldf.utils.utils_tf as utf

import tensorflow as tf
keras = tf.keras
EarlyStopping = keras.callbacks.EarlyStopping
K = keras.backend


def load_image_data(origin_origin_ids, jpgs_map_dict):
    cache_dir_path = os.path.join(os.environ.get("PROJECT_CACHE"), 'load_image_data')
    os.makedirs(cache_dir_path, exist_ok=True)

    origin_origin_ids_hash = hashlib.sha256(json.dumps(origin_origin_ids.tolist()).encode('utf-8')).hexdigest()
    pickle_file_path = Path(os.path.join(cache_dir_path, origin_origin_ids_hash + '.p'))

    if not pickle_file_path.is_file():

        target_rows = 224
        target_cols = 224
        target_dims = 3

        images = []

        n_missing = 0
        N = origin_origin_ids.size
        for i, origin_origin_id in enumerate(origin_origin_ids):

            jpg_filename = f'{origin_origin_id}.jpg'

            jpg_path = ''
            if jpg_filename in jpgs_map_dict:
                jpg_path = jpgs_map_dict[f'{origin_origin_id}.jpg']

            if os.path.isfile(jpg_path):
                # if the file exists, load it!

                outputImage = cv2.imread(jpg_path)

                current_rows = outputImage.shape[0]
                current_cols = outputImage.shape[1]

            else:
                # if the file does not exist,

                print(f'file {jpg_path} does not exist. Using an image of zeros instead.'
                      f"({float(n_missing) / float(i + 1) * 100.:.2f}%) of images "
                      f"missing. {N} total bldgs.")

                outputImage = np.zeros((target_rows, target_cols, target_dims), dtype="uint8")

                current_rows = target_rows
                current_cols = target_cols

                n_missing = n_missing + 1

            diff_rows = current_rows - target_rows
            diff_cols = current_cols - target_cols

            if ((diff_rows < 0) or (diff_cols < 0)):
                # if there are any fewer than the necessary number of rows

                n_missing = n_missing + 1

                if diff_rows < 0:
                    print(
                        f"im row dim {current_rows} below min {target_rows} in "
                        f"path {jpg_path}. {n_missing} of {i}, "
                        f"({float(n_missing) / float(i + 1) * 100.:.2f}%) of images "
                        f"missing. {N} total bldgs.")
                if diff_cols < 0:
                    print(
                        f"im col dim {current_cols} below min {target_cols} in "
                        f"path {jpg_path}. {n_missing} of {i}, "
                        f"({float(n_missing) / float(i + 1) * 100.:.2f}%) of images "
                        f"missing. {N} total bldgs.")

                outputImage = np.zeros((target_rows, target_cols, target_dims), dtype="uint8")
                print(f"sending an image of zeros instead")

            elif ((diff_rows == 0) and (diff_cols == 0)):
                print('all good!')

            else:

                # cropping the output image, ensuring that it aligns in the center of the image
                outputImage = outputImage[int(np.ceil(float(diff_rows) / 2.)):int(-np.floor(float(diff_rows) / 2.)),
                              int(np.ceil(float(diff_cols) / 2.)):int(-np.floor(float(diff_cols) / 2.)), :]

            # import matplotlib.pyplot as plt
            # vizimage = cv2.cvtColor(outputImage, cv2.COLOR_BGR2RGB)
            # plt.imshow(vizimage)
            # plt.show()
            images.append(outputImage)
        images = np.array(images)

        pickle.dump(images, open(pickle_file_path, "wb"), protocol=4)

    else:

        images = pickle.load(open(pickle_file_path, "rb"))

    return images

def save_mbtiles_and_pbf(gdf, output_file_path):
    output_geojson_path = output_file_path + '.geojson'
    gdf.to_file(output_geojson_path, driver="GeoJSON")
    output_mbtiles_path = output_file_path + '.mbtiles'
    run_list = ["tippecanoe",
                "--no-feature-limit", "-z",
                "16", "-Z",
                "1", "-l",
                'default', "-o",
                f"{output_mbtiles_path}", f"{output_geojson_path}"]
    command = ''.join([elem + ' ' for elem in run_list])[:-1]
    print('running ' + command)
    os.system(command)

    output_folder_path = os.path.splitext(output_mbtiles_path)[0]
    run_list = ["tile-join", "--no-tile-size-limit", "-e", f"{output_folder_path}", f"{output_mbtiles_path}"]
    command = ''.join([elem + ' ' for elem in run_list])[:-1]
    print('running ' + command)
    os.system(command)


def copy_web(output_dir, output_gdf):
    # define source and destination folders
    web_source = os.environ.get('OG_WEB_PATH')
    web_destination = os.path.join(output_dir, 'web')

    # define dirs to copy
    dirs_to_copy = os.listdir(web_source)
    if 'data' in dirs_to_copy:
        dirs_to_copy.remove('data')
    if '.git' in dirs_to_copy:
        dirs_to_copy.remove('.git')
    if '.idea' in dirs_to_copy:
        dirs_to_copy.remove('.idea')

    # copy files and dirs
    for elem in dirs_to_copy:
        elem_path = os.path.join(web_source, elem)

        if os.path.isfile(elem_path):
            shutil.copy(elem_path, os.path.join(web_destination, elem))
        else:
            shutil.copytree(elem_path, os.path.join(web_destination, elem))
    print('copied all web files')

    with open(os.path.join(web_destination, 'config.js')) as f:
        contents = f.read()
    contents_str = ''.join(contents).replace('\n', '') \
        .replace('var config =', '') \
        .replace('};', '}') \
        .replace(',}', '}') \
        .replace(' ', '')
    contents_dict = ast.literal_eval(contents_str)

    contents_dict['lat'] = output_gdf.iloc[0]['geometry'].centroid.y
    contents_dict['lon'] = output_gdf.iloc[0]['geometry'].centroid.x
    contents_dict['zoom'] = 13
    contents_dict['bldgs_source_before'] = 'http://localhost:3000/data/output/{z}/{x}/{y}.pbf'
    contents_dict['bldgs_source_before_full'] = 'http://localhost:3000/data/output_run/{z}/{x}/{y}.pbf'

    config_str = 'var config = ' + json.dumps(contents_dict)
    with open(os.path.join(web_destination, 'config.js'), 'w') as f:
        f.write(config_str)
    print('setup config file')


class MultimodalDataGenerator(tf.keras.utils.Sequence):
    """
    Custom data generator class for multimodal vector/image data
    """

    def __init__(self, x_train, s_train_w_hyperparams, jpgs_map_path, batch_size: int = 16):
        self.vector_data = x_train[0]
        self.ids = x_train[1]
        self.s_train_w_hyperparams = s_train_w_hyperparams
        self.batch_size = batch_size
        self.jpgs_map_path = jpgs_map_path

    def __len__(self):
        return np.math.ceil(len(self.vector_data) / self.batch_size)

    def __getitem__(self, index):
        """
        Returns a batch of data
        """

        batch_vector_data = self.vector_data[index * self.batch_size: (index + 1) * self.batch_size]
        batch_ids = self.ids[index * self.batch_size: (index + 1) * self.batch_size]
        batch_image_data = load_image_data(batch_ids, self.jpgs_map_path)

        batch_s_train_w_hyperparams = self.s_train_w_hyperparams[
                                      index * self.batch_size: (index + 1) * self.batch_size]

        return [batch_vector_data, batch_image_data], batch_s_train_w_hyperparams


class LdfMultimodalModel(LdfScalarModel):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        default_kwargs = {
            'save_geojson': True,
            'use_jpgs': True,
            'jpgs_map_path': os.path.join(os.environ.get("PROJECT_OUT"), 'jpgs'),
            'freeze_basemodel_layers': True
        }

        merged_kwargs = {}
        for key in default_kwargs.keys():
            if key in kwargs.keys():
                merged_kwargs[key] = kwargs[key]
            else:
                merged_kwargs[key] = default_kwargs[key]

        self.save_geojson = merged_kwargs['save_geojson']
        self.jpgs_map_path = merged_kwargs['jpgs_map_path']
        self.use_jpgs = merged_kwargs['use_jpgs']
        self.freeze_basemodel_layers = merged_kwargs['freeze_basemodel_layers']

        self.pred_incr = None
        self.posterior_predictive_tail_prob_train = None
        self.posterior_predictive_tail_prob_test = None

    def setup_model(self, X=None, load_model=False):

        self.params = json.loads(self.params)
        self.define_distn()

        self.load_model_ = load_model

        # setup
        if self.gpu == 0:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

        # format output
        out_basename = self.model_name \
                       + f'_{self.error_metric}' \
                       + f'_{datetime.now().strftime("%Y%m%d%H%M%S")}_{self.hash}' \
                       + f'_do{round(self.do_val * 100)}' \
                       + f'_rm{self.reg_mode}' \
                       + f'_rv{round(self.reg_val * 100)}'

        # make output directory
        self.output_dir = os.path.join(self.output_dir, out_basename)
        print(f'output dir: {self.output_dir}')
        os.makedirs(self.output_dir, exist_ok=True)

        args = {key: val for key, val in self.__dict__.items() if
                isinstance(val, int) or
                isinstance(val, str) or
                isinstance(val, float) or
                isinstance(val, bool)}

        # save params to external csv and json files
        with open(os.path.join(self.output_dir, 'args.json'), "w") as outfile:
            json.dump(args, outfile, indent=2)

        # get prob_callback
        self.callbacks_prob_ = \
            dm.get_callbacks_prob(red_lr_on_plateau=self.red_lr_on_plateau,
                                  red_factor=self.red_factor,
                                  red_patience=self.red_patience,
                                  red_min_lr=self.red_min_lr,
                                  verbose=self.verbose,
                                  es_patience=self.es_patience)
        # load model
        if load_model:

            self.model_ = dm.load_model(self.load_json,
                                        self.load_weights,
                                        self.distn)
            print('loaded model')

            self.history_ = None

        else:

            output_dim = 2

            model, _ = \
                dm.define_model(input_dim=X[0].shape[1],
                                output_dim=output_dim,
                                distn=self.distn,
                                do_val=self.do_val,
                                learning_rate_training=self.learning_rate_training,
                                model_name=self.model_name,
                                reg_mode=self.reg_mode,
                                reg_val=self.reg_val,
                                freeze_basemodel_layers=self.freeze_basemodel_layers)

            self.model_ = model

    def fine_tune_model(self,
                        X=None,
                        y=None,
                        X_val=None,
                        y_val=None,
                        use_jpgs=True,
                        batch_size=256,
                        verbose=2,
                        train_epochs=1000,
                        break_out_val=True,
                        use_multiprocessing=False
                        ):

        # set random seed so that images don't need to be constantly recalculated
        import random
        seed = 0
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        tf.keras.utils.set_random_seed(seed)

        s_train_w_hyperparams = y[:, np.newaxis]

        if (use_jpgs and (X_val is None) and break_out_val):
            # setting aside 10% of the training data for validation
            n_train_samps = len(X[0])
            n_es_samps = np.ceil(n_train_samps * 0.1).astype(int)

            # manually splitting training set into training used for fitting, and a 10%
            # portion of the training set used for early stopping.
            indices = np.random.permutation(n_train_samps)
            train_idx, es_idx = indices[:(n_train_samps - n_es_samps)], indices[(n_train_samps - n_es_samps):]
            train_vecs, es_vecs = X[0][train_idx, :], X[0][es_idx, :]
            train_ids, es_ids = X[1][train_idx], X[1][es_idx]

            train_s_train_w_hyperparams, \
                es_s_train_w_hyperparams = \
                s_train_w_hyperparams[train_idx, :], \
                    s_train_w_hyperparams[es_idx, :]

            fit_loader = MultimodalDataGenerator([train_vecs, train_ids],
                                                 train_s_train_w_hyperparams,
                                                 self.jpgs_map_path,
                                                 batch_size=batch_size)
            es_loader = MultimodalDataGenerator([es_vecs, es_ids],
                                                es_s_train_w_hyperparams,
                                                self.jpgs_map_path,
                                                batch_size=batch_size)

            print('fitting model')
            self.history = self.model_.fit(fit_loader,
                                           validation_data=es_loader,
                                           epochs=train_epochs,
                                           verbose=verbose,
                                           callbacks=self.callbacks_prob_)


        elif (use_jpgs and (X_val is None) and (not break_out_val)):

            fit_loader = MultimodalDataGenerator(X,
                                                 s_train_w_hyperparams,
                                                 self.jpgs_map_path,
                                                 batch_size=batch_size)

            print('fitting model')
            self.history = self.model_.fit(fit_loader,
                                           epochs=train_epochs,
                                           verbose=verbose,
                                           callbacks=self.callbacks_prob_)


        elif (use_jpgs and X_val is not None):

            print('asdf')
            s_val_w_hyperparams = y_val[:, np.newaxis]

            fit_loader = MultimodalDataGenerator(X,
                                                 s_train_w_hyperparams,
                                                 self.jpgs_map_path,
                                                 batch_size=batch_size)
            es_loader = MultimodalDataGenerator(X_val,
                                                s_val_w_hyperparams,
                                                self.jpgs_map_path,
                                                batch_size=batch_size)

            print('fitting model')
            self.history = self.model_.fit(fit_loader,
                                           validation_data=es_loader,
                                           epochs=train_epochs,
                                           verbose=verbose,
                                           callbacks=self.callbacks_prob_)

        elif (not use_jpgs and (X_val is None) and break_out_val):

            print('fitting model')
            self.history = self.model_.fit(X,
                                           s_train_w_hyperparams,
                                           validation_split=0.1,
                                           epochs=train_epochs,
                                           verbose=verbose,
                                           callbacks=self.callbacks_prob_)

        elif (not use_jpgs and (X_val is None) and (not break_out_val)):

            print('fitting model')
            self.history = self.model_.fit(X,
                                           s_train_w_hyperparams,
                                           epochs=train_epochs,
                                           verbose=verbose,
                                           callbacks=self.callbacks_prob_)

        else:
            raise Exception("param combination not accounted for in code!")

        # save model definition as .json and model weights as .h5
        model_json = self.model_.to_json()  # serialize model to JSON
        self.output_json_path = os.path.join(self.output_dir, f'model_{self.fold_num_val}_.json')
        self.output_weights_path = os.path.join(self.output_dir, f'model_{self.fold_num_val}_.h5')
        with open(self.output_json_path, 'w') as json_file:
            json_file.write(model_json)
        self.model_.save_weights(self.output_weights_path)  # serialize weights to HDF5
        print('Saved model to disk')

        K.clear_session()

        return self.history.history

    def fit(self,
            X=None,
            y=None,
            X_val=None,
            y_val=None,
            load_model=False,
            use_jpgs=True,
            batch_size=256,
            verbose=2,
            train_epochs=1000
            ):

        self.setup_model(X=X, load_model=load_model)

        if X_val is not None:
            self.fine_tune_model(X=X,
                                 y=y,
                                 X_val=X_val,
                                 y_val=y_val,
                                 use_jpgs=use_jpgs,
                                 batch_size=batch_size,
                                 verbose=verbose,
                                 train_epochs=train_epochs)
        else:
            self.fine_tune_model(X=X,
                                 y=y,
                                 use_jpgs=use_jpgs,
                                 batch_size=batch_size,
                                 verbose=verbose,
                                 train_epochs=train_epochs)

        return self

    def predict(self, X=None):

        print(f'running model')
        n_samps = X[0].shape[0]

        predict_loader = MultimodalDataGenerator(X,
                                                 np.zeros(n_samps)[:, np.newaxis],
                                                 self.jpgs_map_path,
                                                 batch_size=self.batch_size)

        yhat = self.model_.predict(predict_loader, verbose=0)

        mean_preds, \
            std_preds, \
            preds_params, \
            preds_params_flat = \
            self.distn.interpret_predict_output(yhat)

        return mean_preds, std_preds, preds_params, preds_params_flat

    def predict_for_train(self, train_x, train_y, **kwargs):

        self.train_mean_preds, \
            _, \
            self.train_preds_params, \
            self.train_preds_params_flat = \
            self.predict(X=train_x)

        self.mean_train_rmse = None
        self.train_mean_nll = None
        self.train_nlls = None
        if train_y is not None:
            self.train_mean_nll, \
                self.train_nlls = \
                self.distn.compute_nll(self.train_preds_params,
                                       train_y)

            self.mean_train_rmse = us.rmse(train_y, self.train_mean_preds)

        return self

    def predict_for_val(self, val_x, val_y, **kwargs):

        self.val_mean_preds, \
            _, \
            self.val_preds_params, \
            self.val_preds_params_flat = \
            self.predict(X=val_x)

        self.mean_val_rmse = None
        self.val_mean_nll = None
        self.val_nlls = None
        if val_y is not None:
            self.val_mean_nll, \
                self.val_nlls = \
                self.distn.compute_nll(self.val_preds_params,
                                       val_y)

            self.mean_val_rmse = us.rmse(val_y, self.val_mean_preds)

        return self

    def predict_for_test(self, test_x, test_y, **kwargs):

        self.test_mean_preds, \
            _, \
            self.test_preds_params, \
            self.test_preds_params_flat = \
            self.predict(X=test_x)

        self.mean_test_rmse = None
        self.test_cdfs = None
        self.test_mean_nll = None
        self.test_nlls = None
        if test_y is not None:
            self.test_mean_nll, \
                self.test_nlls = \
                self.distn.compute_nll(self.test_preds_params,
                                       test_y)

            self.mean_test_rmse = us.rmse(test_y, self.test_mean_preds)

            self.test_cdfs = self.distn.cdf_params(test_y, self.test_preds_params)

            # evaluate model calibration
            utf.eval_ts_prob_calibration(self.output_dir,
                                         self.test_cdfs,
                                         self.fold_num_test)

        return self

    def predict_for_all(self, x_fit, y):

        self.all_mean_preds, \
            self.all_std_preds, \
            self.all_preds_params, \
            self.all_preds_params_flat = \
            self.predict(X=x_fit)

        self.all_mean_nll, \
            self.all_nlls = \
            self.distn.compute_nll(self.all_preds_params, y)

        self.mean_all_rmse = us.rmse(y, self.all_mean_preds)
        self.mean_all_err = y - self.all_mean_preds

        return self

    def plot_postpred(self,
                      test_inds,
                      x_train,
                      x_test,
                      s_train,
                      s_test):

        import ldf.utils.utils_gapo as ug

        self.pred_incr = \
            ug.plot_posterior_predictive_checks(
                self.output_dir,
                test_inds,
                self.fold_num_val,
                s_test,
                x_test,
                self.mean_test_rmse,
                self.test_preds_params['a_all'],
                self.test_preds_params['b_all'])

        self.posterior_predictive_tail_prob_train, \
            self.posterior_predictive_tail_prob_test = \
            ug.plot_tail_probabilities(
                self.output_dir,
                self.fold_num_val,
                x_train,
                s_train,
                self.train_preds_params['a_all'],
                self.train_preds_params['b_all'],
                x_test,
                s_test,
                self.test_preds_params['a_all'],
                self.test_preds_params['b_all']
            )

    def save_summary(self, data):

        metrics_df = pd.DataFrame(columns=['fold_num_val', 'metric', 'val'], dtype=object)

        # metrics to log for each fold
        metrics_to_log = [
            ('data', data),
            ('a_all_train', self.train_preds_params['a_all']),
            ('b_all_train', self.train_preds_params['b_all']),
            ('a_all_val', self.val_preds_params['a_all']),
            ('b_all_val', self.val_preds_params['b_all']),
            ('a_all_test', self.test_preds_params['a_all']),
            ('b_all_test', self.test_preds_params['b_all']),
            ('train_mean_preds', self.train_mean_preds),
            ('val_mean_preds', self.val_mean_preds),
            ('test_mean_preds', self.test_mean_preds),
            ('mean_train_rmse', self.mean_train_rmse),
            ('mean_val_rmse', self.mean_val_rmse),
            ('mean_test_rmse', self.mean_test_rmse),
            ('train_mean_nll', self.train_mean_nll),
            ('val_mean_nll', self.val_mean_nll),
            ('test_mean_nll', self.test_mean_nll),
            ('history', self.history.history),
            ('pred_incr', self.pred_incr),
            ('posterior_predictive_tail_prob_train', self.posterior_predictive_tail_prob_train),
            ('posterior_predictive_tail_prob_test', self.posterior_predictive_tail_prob_test),
            ('test_cdfs', self.test_cdfs)
        ]

        for metric_name, metric_val in metrics_to_log:
            metric_dict = {
                'fold_num_val': self.fold_num_val,
                'metric': metric_name,
                'val': metric_val
            }
            metrics_df = pd.concat([metrics_df, pd.DataFrame([metric_dict])], ignore_index=True)

        self.metrics_df = metrics_df

        # dump to pickle
        output_path = os.path.join(self.output_dir, 'summary')
        self.metrics_df.to_pickle(output_path + '.p')
        self.metrics_df.to_csv(output_path + '.csv')

        return self.metrics_df

    def save_to_geojson(self, output_gdf, y):

        # log geojson with metrics for each fld:
        output_gdf[f"ldf_mean"] = self.all_mean_preds
        output_gdf[f"ldf_std"] = self.all_std_preds
        output_gdf[f"ldf_std_percent_of_mean"] = self.all_std_preds / self.all_mean_preds * 100.
        output_gdf[f"ldf_err"] = self.mean_all_err
        output_gdf[f"ldf_err_percent_of_mean"] = self.mean_all_err / self.all_mean_preds * 100.
        output_gdf[f"ldf_err_zscore"] = self.mean_all_err / self.all_std_preds
        output_gdf[f"ldf_a"] = self.all_preds_params['a_all']
        output_gdf[f"ldf_b"] = self.all_preds_params['b_all']
        output_gdf[f"ldf_s"] = y
        output_gdf[f"ldf_sample_1"] = self.distn.sample_posterior_params(self.all_preds_params)
        output_gdf[f"ldf_sample_2"] = self.distn.sample_posterior_params(self.all_preds_params)
        output_gdf[f"ldf_sample_3"] = self.distn.sample_posterior_params(self.all_preds_params)

        # save geojson
        if self.save_geojson:
            print('saving geojson, mbtiles, and pbf')
            output_file_path = os.path.join(self.output_dir, 'web', 'data', 'output')
            os.makedirs(output_file_path, exist_ok=True)
            save_mbtiles_and_pbf(output_gdf, output_file_path)
            copy_web(self.output_dir, output_gdf)

        else:
            print('saving csv')
            output_gdf.to_csv(os.path.join(self.output_dir, 'output.csv'))

    def plot_nn_architecture(self):
        from keras.utils.vis_utils import plot_model
        plot_model(self.model_,
                   to_file=os.path.join(self.output_dir, 'model_plot.png'),
                   show_shapes=True, show_layer_names=True)
