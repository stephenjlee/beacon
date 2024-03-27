import sys, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ldf.utils import utils_stat as us

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
sys.path.append(os.environ.get("PROJECT_ROOT"))


def save_geojson(bldgs_df, data, preds_params_all, output_path):
    print('started saving geojson')
    a_of_x = preds_params_all['a_of_x']
    b_of_x = preds_params_all['b_of_x']

    alpha = np.ones_like(a_of_x) * preds_params_all['a0']
    beta = np.ones_like(b_of_x) * preds_params_all['b0']

    y_mean = us.beta_mean(a_of_x + alpha, b_of_x + beta)
    y_std = np.sqrt(us.beta_variance(a_of_x + alpha, b_of_x + beta))

    ids = data['ids']

    nn_res_df = pd.DataFrame.from_dict({
        'origin_origin_id': ids,
        'nn_res_a_of_x': a_of_x,
        'nn_res_b_of_x': b_of_x,
        'nn_res_alpha': alpha,
        'nn_res_beta': beta,
        'nn_res_y_mean': y_mean,
        'nn_res_y_std': y_std,
        'nn_res_y_std_percent_of_mean': y_std / y_mean * 100.,
        'nn_res_y_sample_1_': us.beta_sample(a_of_x + alpha, b_of_x + beta),
        'nn_res_y_sample_2_': us.beta_sample(a_of_x + alpha, b_of_x + beta),
        'nn_res_y_sample_3_': us.beta_sample(a_of_x + alpha, b_of_x + beta),
        'nn_res_y_sample_4_': us.beta_sample(a_of_x + alpha, b_of_x + beta)
    })

    bldgs_df = bldgs_df.set_index('origin_origin_id').join(nn_res_df.set_index('origin_origin_id')).reset_index()

    # after joining, now evaluate the error metric, since we are sure that all columns are ordered correctly
    bldgs_df['nn_res_err'] = bldgs_df['electrified_based_on_rule'].values.astype(float) - bldgs_df[
        'nn_res_y_mean'].values

    output_geojson_path = os.path.join(output_path, 'bldgs_with_ldf.geojson')
    output_mbtiles_path = os.path.join(output_path, 'bldgs_with_ldf.mbtiles')
    os.makedirs(output_path, exist_ok=True)
    bldgs_df.to_file(output_geojson_path, driver="GeoJSON")

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

    print('finished saving buildings geojson')
