import os
import numpy as np
import matplotlib.pyplot as plt
import ldf.utils.utils_stat as us


def posterior_predictive_log_likelihood(y, n, a0, b0, a_of_x, b_of_x, return_sum=True):
    """Compute the posterior predictive likelihood of the data.

    :param y: shape (N, ), observed number of heads
    :param n: shape (N, ), observed number of flips
    :param a_of_x: shape (N, ), pseudo-heads from transformed auxiliary data
    :param b_of_x: shape (N, ), pseudo-tails from transformed auxiliary data
    :param a0: scalar, beta distribution hyperparameter
    :param b0: scalar, beta distribution hyperparameter
    :return: scalar float
    """

    log_p_si_given_xi = us.betabinom_logpmf(y, a0 + a_of_x, b0 + b_of_x, n)

    if return_sum:
        return np.sum(log_p_si_given_xi)
    else:
        return log_p_si_given_xi


def credible_region(p, gamma=0.9):
    """Compute a minimal set of indices containing at least gamma probability mass.

    :param p: a discrete probability mass function
    :param gamma: the cumulative probability mass desired
    :return: a minimal set of indices s.t. sum_i p[i] >= gamma
    """
    cix = np.flip(np.argsort(p), axis=0)
    # set of the indices totalling at least gamma
    return cix[:np.where(np.cumsum(p[cix]) > gamma)[0][0] + 1]


def plot_posterior_predictive_checks(test_indices,
                                     fold,
                                     y,
                                     n,
                                     params,
                                     preds_params,
                                     gamma=0.9,
                                     plot=False,
                                     output_dir=None):
    """

    :param test_indices: shape (N, ), contains int indices of test points for plotting
    :param fold: scalar int, used for output/displaying
    :param y: shape (N, ), observed number of heads
    :param n: shape (N, ), observed number of flips
    :param a_of_x: shape (N, ), pseudo-heads from transformed auxiliary data
    :param b_of_x: shape (N, ), pseudo-tails from transformed auxiliary data
    :param a0: scalar, beta distribution hyperparameter
    :param b0: scalar, beta distribution hyperparameter
    :param gamma: the amount of probability mass in the credible region
    :param plot: boolean (default False)
    :param output_dir: where to save plots (default None)
    :return: shape (N, ), whether or not the observed point was in the credible region
    """

    a_of_x = preds_params['a_of_x']
    b_of_x = preds_params['b_of_x']
    a0 = params['a0']
    b0 = params['b0']

    n = n.astype(int)  # convert in case it's a float

    # Plot the posterior predictive checks for the heldout survey data
    if plot:
        plot_shape = (6, 7)
        # i_to_plot = np.arange(plot_shape[0] * plot_shape[1])
        i_to_plot = np.sort(np.random.choice(y.size, plot_shape[0] * plot_shape[1]), axis=None)
        plt.figure(figsize=(16, 8))
        print('plotting predictive probability figures')
        print(f'fold number: {fold}')

    pred_incr = np.zeros(y.size)
    for i in range(y.size):

        # compute log P(Y = y | x; \phi) for y = 0, 1, ..., n[i]
        y_eval = np.arange(n[i] + 1)
        p_yi_given_xi = np.exp(us.betabinom_logpmf(
            y_eval, a0 + a_of_x[i], b0 + b_of_x[i], n[i]))

        # compute the gamma*100% credible region of minimum width and check
        # whether the observed value is included in it.
        cr = credible_region(p_yi_given_xi, gamma=gamma)
        pred_incr[i] = y[i] in cr

        if plot:
            if np.isin(i, i_to_plot):

                plt.subplot(plot_shape[0], plot_shape[1],
                            1 + np.where(i_to_plot == i)[0][0])
                if pred_incr[i]:
                    color_str = 'C2'
                else:
                    color_str = 'C3'
                plt.bar(y_eval[:n[i]], p_yi_given_xi[:n[i]], width=1.0,
                        alpha=0.25, color='C0',
                        label=r'${}$'.format(test_indices[i]))
                ylim = plt.ylim()
                plt.plot(np.tile(y[i], 2), np.array(plt.ylim()), color='C0', linewidth=2)
                plt.grid()
                plt.legend(loc='upper right', facecolor=color_str, framealpha=0.33,
                           handlelength=0, handletextpad=0, fancybox=True,
                           fontsize='x-large')
                plt.axis([plt.xlim()[0], plt.xlim()[1], ylim[0], ylim[1]])

    if plot:
        plt.suptitle(
            r'$p(y \:|\: x;\, \hat{{\phi}})$: Calibration = {:.3g}% (% $\in$ CR$_{{{:g}}}$)'.format(
                100 * np.mean(pred_incr), 100 * gamma),
            fontsize='xx-large')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig('{}/fold_{}_ppc.pdf'.format(output_dir, fold))

    return pred_incr


def plot_posterior_predictive_checks_orig(test_indices,
                                          fold,
                                          y,
                                          n,
                                          a_of_x,
                                          b_of_x,
                                          a0,
                                          b0,
                                          gamma=0.9,
                                          plot=False,
                                          output_dir=None):
    """

    :param test_indices: shape (N, ), contains int indices of test points for plotting
    :param fold: scalar int, used for output/displaying
    :param y: shape (N, ), observed number of heads
    :param n: shape (N, ), observed number of flips
    :param a_of_x: shape (N, ), pseudo-heads from transformed auxiliary data
    :param b_of_x: shape (N, ), pseudo-tails from transformed auxiliary data
    :param a0: scalar, beta distribution hyperparameter
    :param b0: scalar, beta distribution hyperparameter
    :param gamma: the amount of probability mass in the credible region
    :param plot: boolean (default False)
    :param output_dir: where to save plots (default None)
    :return: shape (N, ), whether or not the observed point was in the credible region
    """

    n = n.astype(int)  # convert in case it's a float

    # Plot the posterior predictive checks for the heldout survey data
    if plot:
        plot_shape = (6, 7)
        # i_to_plot = np.arange(plot_shape[0] * plot_shape[1])
        i_to_plot = np.sort(np.random.choice(y.size, plot_shape[0] * plot_shape[1]), axis=None)
        plt.figure(figsize=(16, 8))
        print('plotting predictive probability figures')
        print(f'fold number: {fold}')

    pred_incr = np.zeros(y.size)
    for i in range(y.size):

        # compute log P(Y = y | x; \phi) for y = 0, 1, ..., n[i]
        y_eval = np.arange(n[i] + 1)
        p_yi_given_xi = np.exp(us.betabinom_logpmf(
            y_eval, a0 + a_of_x[i], b0 + b_of_x[i], n[i]))

        # compute the gamma*100% credible region of minimum width and check
        # whether the observed value is included in it.
        cr = credible_region(p_yi_given_xi, gamma=gamma)
        pred_incr[i] = y[i] in cr

        if plot:
            if np.isin(i, i_to_plot):

                plt.subplot(plot_shape[0], plot_shape[1],
                            1 + np.where(i_to_plot == i)[0][0])
                if pred_incr[i]:
                    color_str = 'C2'
                else:
                    color_str = 'C3'
                plt.bar(y_eval[:n[i]], p_yi_given_xi[:n[i]], width=1.0,
                        alpha=0.25, color='C0',
                        label=r'${}$'.format(test_indices[i]))
                ylim = plt.ylim()
                plt.plot(np.tile(y[i], 2), np.array(plt.ylim()), color='C0', linewidth=2)
                plt.grid()
                plt.legend(loc='upper right', facecolor=color_str, framealpha=0.33,
                           handlelength=0, handletextpad=0, fancybox=True,
                           fontsize='x-large')
                plt.axis([plt.xlim()[0], plt.xlim()[1], ylim[0], ylim[1]])

    if plot:
        plt.suptitle(
            r'$p(y \:|\: x;\, \hat{{\phi}})$: Calibration = {:.3g}% (% $\in$ CR$_{{{:g}}}$)'.format(
                100 * np.mean(pred_incr), 100 * gamma),
            fontsize='xx-large')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig('{}/fold_{}_ppc.pdf'.format(output_dir, fold))

    return pred_incr


def plot_posterior_predictive_and_metrics(test_indices,
                                          fold,
                                          y,
                                          params,
                                          preds_params,
                                          output_dir=None):
    a_of_x = preds_params['a_of_x']
    b_of_x = preds_params['b_of_x']
    a0 = params['a0']
    b0 = params['b0']

    """
    :param test_indices: shape (N, ), contains int indices of test points for plotting
    :param fold: scalar int, used for output/displaying
    :param y: shape (N, ), observed number of heads
    :param n: shape (N, ), observed number of flips
    :param a_of_x: shape (N, ), pseudo-heads from transformed auxiliary data
    :param b_of_x: shape (N, ), pseudo-tails from transformed auxiliary data
    :param a0: scalar, beta distribution hyperparameter
    :param b0: scalar, beta distribution hyperparameter
    :param gamma: the amount of probability mass in the credible region
    :param plot: boolean (default False)
    :param output_dir: where to save plots (default None)
    :return: shape (N, ), whether or not the observed point was in the credible region
    """
    n = 100
    plot_shape = (100, 2)

    a = a0 + a_of_x
    b = b0 + b_of_x
    betabinom_distn_mean_n1 = us.betabinom_mean(a, b, n)
    betabinom_distn_var_n1 = us.betabinom_variance(a, b, n)
    betabinom_distn_std_n1 = np.sqrt(betabinom_distn_var_n1)

    indices_by_std = np.argsort(betabinom_distn_std_n1)
    indices_to_keep = np.linspace(0, a.size - 1, num=plot_shape[0]).astype(int)
    i_to_plot = indices_by_std[indices_to_keep]

    # i_to_plot = np.arange(plot_shape[0] * plot_shape[1])
    # i_to_plot = np.sort(np.random.choice(y.size, plot_shape[0] * plot_shape[1]), axis=None)
    fig, ax_orig = plt.subplots(plot_shape[0], plot_shape[1])
    fig.set_figheight(plot_shape[0] * 2)
    fig.set_figwidth(5)
    print('plotting predictive probability figures')
    print(f'fold number: {fold}')

    for ii, i in enumerate(i_to_plot):

        # plot beta-binomial distribution with n1
        j = 1
        ax = plt.subplot(plot_shape[0], plot_shape[1], j + ii * plot_shape[1])
        beta_binom = np.exp(us.betabinom_logpmf(
            np.arange(n + 1), a[i], b[i], n))
        ax.plot(np.arange(n + 1), beta_binom)
        ylim = plt.ylim()
        if ylim[1] - ylim[0] < 1e-10:
            plt.ylim((ylim[0] - 1e-10, ylim[1] + 1e-10))
        ax.set_title(f'beta-binom n={n}', fontsize=10)
        ax.set_xlabel('x')
        ax.set_ylabel('pmf')

        # plot horizontal bar chart with mean and var
        j = j + 1
        ax = plt.subplot(plot_shape[0], plot_shape[1], j + ii * plot_shape[1])
        ax.barh(2, betabinom_distn_std_n1[i], left=betabinom_distn_mean_n1[i] - 0.5 * betabinom_distn_std_n1[i],
                height=1,
                align='center')
        ax.plot(betabinom_distn_mean_n1[i], 2, 'wo', markersize=15)
        ax.plot(betabinom_distn_mean_n1[i], 2, 'bo', markersize=10)
        ax.set_ylim(0, 4)
        ax.set_xlim(0, n)
        ax.get_yaxis().set_visible(False)

        if plot_shape[0] * plot_shape[1] == j + ii * plot_shape[1]:
            break

    # plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.tight_layout()
    plt.savefig('{}/fold_{}_pp.pdf'.format(output_dir, fold))

    return


def plot_posterior_predictive_and_metrics_orig(test_indices,
                                               fold,
                                               y,
                                               a_of_x,
                                               b_of_x,
                                               a0,
                                               b0,
                                               output_dir=None):
    """
    :param test_indices: shape (N, ), contains int indices of test points for plotting
    :param fold: scalar int, used for output/displaying
    :param y: shape (N, ), observed number of heads
    :param n: shape (N, ), observed number of flips
    :param a_of_x: shape (N, ), pseudo-heads from transformed auxiliary data
    :param b_of_x: shape (N, ), pseudo-tails from transformed auxiliary data
    :param a0: scalar, beta distribution hyperparameter
    :param b0: scalar, beta distribution hyperparameter
    :param gamma: the amount of probability mass in the credible region
    :param plot: boolean (default False)
    :param output_dir: where to save plots (default None)
    :return: shape (N, ), whether or not the observed point was in the credible region
    """
    n = 100
    plot_shape = (100, 2)

    a = a0 + a_of_x
    b = b0 + b_of_x
    betabinom_distn_mean_n1 = us.betabinom_mean(a, b, n)
    betabinom_distn_var_n1 = us.betabinom_variance(a, b, n)
    betabinom_distn_std_n1 = np.sqrt(betabinom_distn_var_n1)

    indices_by_std = np.argsort(betabinom_distn_std_n1)
    indices_to_keep = np.linspace(0, a.size - 1, num=plot_shape[0]).astype(int)
    i_to_plot = indices_by_std[indices_to_keep]

    # i_to_plot = np.arange(plot_shape[0] * plot_shape[1])
    # i_to_plot = np.sort(np.random.choice(y.size, plot_shape[0] * plot_shape[1]), axis=None)
    fig, ax_orig = plt.subplots(plot_shape[0], plot_shape[1])
    fig.set_figheight(plot_shape[0] * 2)
    fig.set_figwidth(5)
    print('plotting predictive probability figures')
    print(f'fold number: {fold}')

    for ii, i in enumerate(i_to_plot):

        # plot beta-binomial distribution with n1
        j = 1
        ax = plt.subplot(plot_shape[0], plot_shape[1], j + ii * plot_shape[1])
        beta_binom = np.exp(us.betabinom_logpmf(
            np.arange(n + 1), a[i], b[i], n))
        ax.plot(np.arange(n + 1), beta_binom)
        ylim = plt.ylim()
        if ylim[1] - ylim[0] < 1e-10:
            plt.ylim((ylim[0] - 1e-10, ylim[1] + 1e-10))
        ax.set_title(f'beta-binom n={n}', fontsize=10)
        ax.set_xlabel('x')
        ax.set_ylabel('pmf')

        # plot horizontal bar chart with mean and var
        j = j + 1
        ax = plt.subplot(plot_shape[0], plot_shape[1], j + ii * plot_shape[1])
        ax.barh(2, betabinom_distn_std_n1[i], left=betabinom_distn_mean_n1[i] - 0.5 * betabinom_distn_std_n1[i],
                height=1,
                align='center')
        ax.plot(betabinom_distn_mean_n1[i], 2, 'wo', markersize=15)
        ax.plot(betabinom_distn_mean_n1[i], 2, 'bo', markersize=10)
        ax.set_ylim(0, 4)
        ax.set_xlim(0, n)
        ax.get_yaxis().set_visible(False)

        if plot_shape[0] * plot_shape[1] == j + ii * plot_shape[1]:
            break

    # plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.tight_layout()
    plt.savefig('{}/fold_{}_pp.pdf'.format(output_dir, fold))

    return


def plot_posterior_samples(y,
                           n,
                           params,
                           preds_params,
                           output_dir=None):
    fold = 0
    a_of_x = preds_params['a_of_x']
    b_of_x = preds_params['b_of_x']
    a0 = params['a0']
    b0 = params['b0']

    """

    :param test_indices: shape (N, ), contains int indices of test points for plotting
    :param fold: scalar int, used for output/displaying
    :param y: shape (N, ), observed number of heads
    :param n: shape (N, ), observed number of flips
    :param a_of_x: shape (N, ), pseudo-heads from transformed auxiliary data
    :param b_of_x: shape (N, ), pseudo-tails from transformed auxiliary data
    :param a0: scalar, beta distribution hyperparameter
    :param b0: scalar, beta distribution hyperparameter
    :param gamma: the amount of probability mass in the credible region
    :param plot: boolean (default False)
    :param output_dir: where to save plots (default None)
    :return: shape (N, ), whether or not the observed point was in the credible region
    """

    n = n.astype(int)  # convert in case it's a float
    step_size = 0.001
    x = np.arange(step_size, 1. - step_size, step_size)

    # Plot the posterior for the heldout survey data
    plot_shape = (6, 7)
    i_to_plot = np.sort(np.random.choice(y.size, plot_shape[0] * plot_shape[1]), axis=None)

    plt.figure(figsize=(16, 8))
    print('plotting predictive probability figures')
    print(f'fold number: {fold}')

    for i in range(y.size):

        if np.isin(i, i_to_plot):
            plt.subplot(plot_shape[0], plot_shape[1],
                        1 + np.where(i_to_plot == i)[0][0])

            mean_val = (a0 + a_of_x[i]) / (a0 + a_of_x[i] + b0 + b_of_x[i])

            if (y[i] <= 0.5 and mean_val <= 0.5) or \
                    (y[i] >= 0.5 and mean_val >= 0.5):
                color_str = 'C3'
            else:
                color_str = 'C2'

            theta = np.exp(us.beta_logpdf(x, a0 + a_of_x[i], b0 + b_of_x[i]))
            plt.plot(x, theta, label=f'Dam: {bool(y[i])}')
            ylim = plt.ylim()
            plt.plot([mean_val, mean_val], [ylim[0], ylim[1]])
            plt.grid()
            plt.legend(loc='upper right',
                       facecolor=color_str,
                       framealpha=0.33,
                       handlelength=0,
                       handletextpad=0,
                       fancybox=True,
                       fontsize='x-large')
            plt.axis([plt.xlim()[0], plt.xlim()[1], ylim[0], ylim[1]])

    plt.suptitle(
        r'$p(theta \:|\: x;\, \hat{{\phi}})$',
        fontsize='xx-large')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(output_dir, f'fold_{fold}_posterior.pdf')
    plt.savefig(save_path)
    print(f'saved figure: {save_path}')

    return


def plot_posterior_samples_orig(fold,
                                y,
                                n,
                                a_of_x,
                                b_of_x,
                                a0,
                                b0,
                                plot=True,
                                output_dir=None):
    """

    :param test_indices: shape (N, ), contains int indices of test points for plotting
    :param fold: scalar int, used for output/displaying
    :param y: shape (N, ), observed number of heads
    :param n: shape (N, ), observed number of flips
    :param a_of_x: shape (N, ), pseudo-heads from transformed auxiliary data
    :param b_of_x: shape (N, ), pseudo-tails from transformed auxiliary data
    :param a0: scalar, beta distribution hyperparameter
    :param b0: scalar, beta distribution hyperparameter
    :param gamma: the amount of probability mass in the credible region
    :param plot: boolean (default False)
    :param output_dir: where to save plots (default None)
    :return: shape (N, ), whether or not the observed point was in the credible region
    """

    n = n.astype(int)  # convert in case it's a float
    step_size = 0.001
    x = np.arange(step_size, 1. - step_size, step_size)

    # Plot the posterior for the heldout survey data
    plot_shape = (6, 7)
    i_to_plot = np.sort(np.random.choice(y.size, plot_shape[0] * plot_shape[1]), axis=None)

    plt.figure(figsize=(16, 8))
    print('plotting predictive probability figures')
    print(f'fold number: {fold}')

    for i in range(y.size):

        if np.isin(i, i_to_plot):
            plt.subplot(plot_shape[0], plot_shape[1],
                        1 + np.where(i_to_plot == i)[0][0])

            mean_val = (a0 + a_of_x[i]) / (a0 + a_of_x[i] + b0 + b_of_x[i])

            if (y[i] <= 0.5 and mean_val <= 0.5) or \
                    (y[i] >= 0.5 and mean_val >= 0.5):
                color_str = 'C3'
            else:
                color_str = 'C2'

            theta = np.exp(us.beta_logpdf(x, a0 + a_of_x[i], b0 + b_of_x[i]))
            plt.plot(x, theta, label=f'Dam: {bool(y[i])}')
            ylim = plt.ylim()
            plt.plot([mean_val, mean_val], [ylim[0], ylim[1]])
            plt.grid()
            plt.legend(loc='upper right',
                       facecolor=color_str,
                       framealpha=0.33,
                       handlelength=0,
                       handletextpad=0,
                       fancybox=True,
                       fontsize='x-large')
            plt.axis([plt.xlim()[0], plt.xlim()[1], ylim[0], ylim[1]])

    plt.suptitle(
        r'$p(theta \:|\: x;\, \hat{{\phi}})$',
        fontsize='xx-large')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(output_dir, f'fold_{fold}_posterior.pdf')
    plt.savefig(save_path)
    print(f'saved figure: {save_path}')

    return
