import os
import matplotlib.pyplot as plt

def plot_hist(hist, output_dir, rmse='', nll=''):
    fig = plt.figure()
    plt.plot(hist.history['loss'], label='train')
    plt.plot(hist.history['val_loss'], label='val')
    plt.legend()
    plt.yscale('symlog')
    plt.title(f'Test NLL: {nll} \n Test RMSE: {rmse} ')
    plt.savefig(os.path.join(output_dir, 'loss_vs_epoch.pdf'))
    plt.close(fig)