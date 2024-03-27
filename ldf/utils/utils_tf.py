import sys, os

os.environ["OMP_NUM_THREADS"] = "1"
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import model_from_json

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
sys.path.append(os.environ.get("PROJECT_ROOT"))

def load_saved_model(json_path, weights_path):
    with open(json_path, 'r') as f:
        model = model_from_json(f.read())  # load the model
    model.load_weights(weights_path)
    return model

def eval_ts_prob_calibration(output_dir, cdfs, fold):

    fig = plt.figure()
    plt.hist(cdfs, bins=50)
    plt.title(f'Model Calibration Histogram for Fold {fold}')
    plt.xlabel('Cumulative Distribution Function Evaluations \n at Observed Consumption Values')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'calibration.pdf'))
    plt.close(fig)

    return cdfs
