import numpy as np
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import backend as K

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

import ldf.models.nn_architectures as nn_architectures
import ldf.utils.utils_tf as utf
from ldf.models.distn_gapo import DistnGapo

epsilon = np.finfo(np.float32).eps

def get_callbacks_prob(red_lr_on_plateau=True,
                       red_factor=0.1,
                       red_patience=10,
                       red_min_lr=0.00000001,
                       verbose=2,
                       es_patience=20):

    # define callback for early stopping
    callbacks_prob = []

    if red_lr_on_plateau:
        callbacks_prob.append(ReduceLROnPlateau(
            factor=red_factor,
            patience=red_patience,
            min_lr=red_min_lr))

    if not es_patience == -1:
        callbacks_prob.append(EarlyStopping(
            monitor='val_loss',
            mode='min',
            verbose=verbose,
            patience=es_patience,
            restore_best_weights=True))

    return callbacks_prob


def get_callbacks_mse(verbose=2,
                      es_patience_for_transfer_learning_model=20):

    # define callback for early stopping
    callbacks_mse = []

    callbacks_mse.append(EarlyStopping(
        monitor='val_loss',
        mode='min',
        verbose=verbose,
        patience=es_patience_for_transfer_learning_model,
        restore_best_weights=True))

    return callbacks_mse


def mean_squared_error_tf(y_true, y_pred):  # 2 dimensional y_true and y_pred

    if len(y_true.shape) > 2:
        y_true = y_true[:, :, 0]
        y_pred = y_pred[:, :, 0]

    return K.mean(K.square(y_pred - y_true), axis=-1)

def define_model(input_dim=None,
                 output_dim=None,
                 distn=None,
                 do_val=0.0,
                 learning_rate_training=0.0001,
                 reg_mode='L2',
                 reg_val=0.00,
                 model_name='XLARGE',
                 freeze_basemodel_layers=True):

    model_mse = None

    if model_name == 'XLARGE':
        model_prob = nn_architectures.define_xlarge_nn_model(input_dim, output_dim, do_val, reg_mode, reg_val)
    elif model_name == 'LARGER':
        model_prob = nn_architectures.define_large_nn_model(input_dim, output_dim, do_val, reg_mode, reg_val)
    elif model_name == 'SHALLOW':
        model_prob = nn_architectures.define_shallow_nn_model(input_dim, output_dim, do_val, reg_mode, reg_val)
    elif model_name == 'LARGER_RESNET':
        model_prob = nn_architectures.get_larger_resnet(input_dim,
                                                        reg_mode,
                                                        reg_val,
                                                        do_val,
                                                        freeze_basemodel_layers=freeze_basemodel_layers)
    else:
        raise Exception(f"Could not find model_name: {model_name}")

    if isinstance(distn, DistnGapo):
        model_prob.compile(loss=distn.tf_nll,
                           optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate_training),
                           metrics=[distn.tf_rmse])
    else:
        model_prob.compile(loss=distn.tf_nll,
                           optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate_training))

    return model_prob, model_mse


def load_model(load_json,
               load_weights,
               distn,
               learning_rate_training=0.0001):
    model_prob = utf.load_saved_model(load_json, load_weights)

    model_prob.compile(loss=distn.tf_nll,
                       optimizer=tf.keras.optimizers.Adam(
                           learning_rate=learning_rate_training)
                       )

    return model_prob
