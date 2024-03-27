import os
from dotenv import load_dotenv, find_dotenv

import numpy as np
import matplotlib.pyplot as plt
import ldf.utils.utils_stat as us

load_dotenv(find_dotenv())  # load environment variables


def credible_region(p, gam=0.9):
    cix = np.flip(np.argsort(p), axis=0)
    # set of the indices totalling at least gamma
    return cix[:np.where(np.cumsum(p[cix]) > gam)[0][0] + 1]


def posterior_predictive_log_likelihood(x, alpha, beta, s,
                                        return_sum=True, phi_eval=None, add_a=None, add_b=None):
    """Compute the posterior predictive likelihood

    :param x:  shape (N, D)
    :param alpha: shape (N, ). These are the hyperparameters on alpha
    :param beta: shape (N, ). These are the hyperparameters on beta
    :param s: shape (N, ). The ground truth counts
    :param return_sum: Boolean
    :param phi_eval: shape (D, ). Optional. If phi_eval is passed, then pseudocounts will be calcualted using it and added to alpha.
    :param add_a: shape (D, ). These are additional
    :param add_b: shape (D, )
    :return: scalar float
    """
    alpha_all = alpha
    beta_all = beta

    if add_a is not None and add_b is not None:
        alpha_all = alpha_all + add_a
        beta_all = beta_all + add_b

    log_p_si_given_xi = us.gammapoisson_logpmf(s, alpha_all, beta_all)

    if return_sum:
        return np.sum(log_p_si_given_xi)
    else:
        return log_p_si_given_xi


def posterior_predictive_tail_probabilities(x, alpha, beta, s, phi_eval=None, add_a=None, add_b=None):
    """Compute the posterior predictive tail probabilities: Pr{Y > s_i}
    for
        Y ~ P(y_i | x_i; \phi)
          ~ GammaPoisson(y_i; alpha + sum(a(x; phi)), beta + sum(b(x; phi)))

          Here this amounts to I_{1/(1+b)}(y+1, a) where
          I_x(alpha, beta) is the incomplete beta function

    :param phi_eval: shape (D, )
    :param x:  shape (N, D)
    :param alpha: shape (N, )
    :param beta: shape (N, )
    :param s: shape (N, )
    :return: scalar float
    """
    import scipy.special as sc

    alpha_all = alpha
    beta_all = beta

    if add_a is not None and add_b is not None:
        alpha_all = alpha_all + add_a
        beta_all = beta_all + add_b

    return sc.betainc(s + 1, alpha_all, 1 / (1 + beta_all))  # I_{z}(x, y)


def plot_posterior_predictive_checks(
        OUTPUT_FOLDER_NAME,
        test_indices,
        fold,
        s_test,
        x_test,
        rmse_mean_test,
        a_all,
        b_all):

    # Plot the posterior predictive checks for the heldout survey data
    plot_shape = (6, 7)
    i_to_plot = np.arange(plot_shape[0] * plot_shape[1])

    thresh = 1.0
    cr_gam = 0.9
    pred_zscore = np.zeros(test_indices.size)
    pred_incr = np.zeros(test_indices.size)
    plt.figure(figsize=(16, 8))

    print('plotting predictive probability figures')
    print('fold number: {}'.format(str(fold)))

    for i, (test_ind, a, b) in enumerate(zip(test_indices, a_all, b_all)):

        print(f'{i} of {test_indices.size}')

        # compute the log posterior predictive probability for s = 0, 1, ..., eta[i]
        num_eval_pts = int(max(s_test[i] * 10, 200000))
        s_eval = np.arange(num_eval_pts)
        log_p_si_given_xi = posterior_predictive_log_likelihood(
            np.tile(x_test[i, :], (num_eval_pts, 1)),
            np.tile(a, num_eval_pts),
            np.tile(b, num_eval_pts),
            s_eval, return_sum=False, phi_eval=None)

        p_si_given_xi = np.exp(log_p_si_given_xi)

        if len(np.where(np.cumsum(p_si_given_xi) > 0.9999)[0]) == 0:
            print(
                "The following error means you need to increase num_eval_pts:")
            # this will only get printed if the next line throws an error
            # because cumsum(p_si_given_xi) never hits 0.9999 on the specified range
        max_k = max(np.where(np.cumsum(p_si_given_xi) > 0.9999)[0][0], 10)
        max_k = int(max(max_k, 3 * s_test[i]))

        # compute moments
        mean_si = np.dot(p_si_given_xi, s_eval)
        std_si = np.sqrt(np.dot(p_si_given_xi, s_eval ** 2) - mean_si ** 2)
        pred_zscore[i] = (s_test[i] - mean_si) / std_si

        cr = credible_region(p_si_given_xi, gam=cr_gam)
        pred_incr[i] = s_test[i] in cr

        if np.isin(i, i_to_plot):
            plt.subplot(plot_shape[0], plot_shape[1],
                        1 + np.where(i_to_plot == i)[0][0])
            if pred_incr[i]:
                color_str = 'C2'
            else:
                color_str = 'C3'
            plt.bar(s_eval[:max_k], p_si_given_xi[:max_k], width=1.0,
                    alpha=0.25, color='C0',
                    label=r'${}$'.format(test_indices[i]))
            ylim = plt.ylim()
            plt.plot(np.tile(s_test[i], 2), np.array(plt.ylim()), color='C0',
                     linewidth=2)
            plt.grid()
            plt.legend(loc='upper right', facecolor=color_str, framealpha=0.33,
                       handlelength=0, handletextpad=0, fancybox=True,
                       fontsize='x-large')
            plt.axis([plt.xlim()[0], plt.xlim()[1], ylim[0], ylim[1]])
    #
    plt.suptitle(
        r'$p(s \:|\: x;\, \hat{{\phi}})$: RMSE = {:.3g}, Calibration = {:.3g}% (% $\in$ CR$_{{{:g}}}$)'.format(
            rmse_mean_test, 100 * np.mean(pred_incr), 100 * cr_gam),
        fontsize='xx-large')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(OUTPUT_FOLDER_NAME, 'fold_{}_posterior_predictive_checks.pdf'.format(fold)))

    print('PL Avg Zscore: {}'.format(np.mean(pred_zscore)))
    print('PL Pct in CR90: {}%'.format(100 * np.mean(pred_incr)))

    return pred_incr

def plot_tail_probabilities(OUTPUT_FOLDER_NAME,
                            fold,
                            x_train,
                            s_train,
                            a_all_train,
                            b_all_train,
                            x_test,
                            s_test,
                            a_all_test,
                            b_all_test, ):
    posterior_predictive_tail_prob_train = posterior_predictive_tail_probabilities(
        x_train,
        a_all_train,
        b_all_train,
        s_train)
    posterior_predictive_tail_prob_test = posterior_predictive_tail_probabilities(
        x_test,
        a_all_test,
        b_all_test,
        s_test)

    plt.figure()
    plt.subplot(121)
    plt.hist(posterior_predictive_tail_prob_train, bins=21, range=[0, 1])
    plt.grid()
    plt.xlabel('Train')
    plt.subplot(122)
    plt.hist(posterior_predictive_tail_prob_test, bins=21, range=[0, 1])
    plt.grid()
    plt.xlabel('Test')
    plt.savefig(os.path.join(OUTPUT_FOLDER_NAME, 'fold_{}_predictive_tail_probs.pdf'.format(fold)))

    return posterior_predictive_tail_prob_train, posterior_predictive_tail_prob_test
