import json

from tensorflow.keras import backend as K
import tensorflow_probability as tfp
import tensorflow as tf
import numpy as np

from scipy.stats import nbinom

from ldf.models.distn_base import DistnBase
import ldf.utils.utils_stat as us


class DistnBetaBinom(DistnBase):
    def __init__(self, params):
        super().__init__(params)
        self.a0 = params['a0']
        self.b0 = params['b0']

    def get_distn_type(self):
        return 'discrete'

    @staticmethod
    def np_ll(x, a, b, n):
        # define the log likelihood function via numpy
        return us.betabinom_logpmf(x, a, b, n)

    def tf_nll(self, yn_true, y_pred):  # 2 dimensional y_true and y_pred
        """Evaluate the beta-binomial log-likelihood pmf function.

        Note that yn_true and y_pred contain a variety of different things,
        corresponding to the things invariant to the network,
            yn_true := [y_i; n_i]
        and the things changed by the network,
            y_pred :=  [a(x_i); b(x_i)].

        :param yn_true: shape [N, 2], contains the info that isn't impacted by
                                     the network, the [y_i; n_i]
        :param y_pred: shape [N, 2], contains the output of the network, namely
                                     the transformed value of x into the pseudo-
                                     heads and pseudo-tails; [a(x_i); b(x_i)]
        :return: scalar, NLL
            = - MEAN(log P(y_i | x_i))
            = - MEAN(log BetaBin(y_i | n_i, a0 + a(x_i), b0 + b(x_i)))
        """
        concentration = y_pred + tf.constant([self.a0, self.b0])  # shape [N, 2]
        total_counts = yn_true[:, 1]  # shape [N]
        dist = tfp.distributions.DirichletMultinomial(total_counts, concentration)
        counts = tf.stack([yn_true[:, 0],
                           yn_true[:, 1] - yn_true[:, 0]], axis=1)  # shape [N, 2]

        return -tf.reduce_mean(dist.log_prob(counts))


    def interpret_predict_output_ts(self, yhat, n):

        a_of_x = yhat[:, 0]
        b_of_x = yhat[:, 1]


        mean_preds = us.betabinom_mean(self.a0 + a_of_x,
                                     self.b0 + b_of_x,
                                     n)

        preds_params = {
            'a_of_x': a_of_x,
            'b_of_x': b_of_x,
            'a0': self.a0,
            'b0': self.b0
        }

        preds_params_flat = []
        for a, b in zip(a_of_x, b_of_x):
            preds_params_flat.append({
                'a_of_x': a,
                'b_of_x': b,
                'a0': self.a0,
                'b0': self.b0
            })

        return mean_preds, preds_params, preds_params_flat

    def compute_nll(self, preds_params, y, n):
        a_of_x = preds_params['a_of_x']
        b_of_x = preds_params['b_of_x']

        nlls = -1. * self.np_ll(y, a_of_x + self.a0, b_of_x + self.b0, n)
        mean_nll = np.mean(nlls)

        return mean_nll, nlls
