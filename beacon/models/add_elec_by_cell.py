import os, sys, argparse, json, pickle, multiprocessing, random
import numpy as np
import geopandas as gpd

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
sys.path.append(os.environ.get("PROJECT_ROOT"))

seed = 0
random.seed(seed)
np.random.seed(seed)
rng = np.random.default_rng()


def parse_args(args):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Get shapefile in gridded form for country.')

    parser.add_argument('-cc', '--country_code',
                        default='',
                        help='The ISO country code for the country being analyzed')
    parser.add_argument('-pps', '--par_pool_size', type=int, default=12,
                        help='parallel pool size for cuda processing')
    parser.add_argument('-ekp', '--elec_keys_path',
                        default=os.path.join(os.environ.get("PROJECT_DATA"),
                                             'trained_models',
                                             'temp',
                                             'data_keys.json'), help='')
    parser.add_argument('-esp', '--elec_scaler_path',
                        default=os.path.join(os.environ.get("PROJECT_DATA"),
                                             'trained_models',
                                             'temp',
                                             'scaler.pkl'), help='')
    parser.add_argument('-elp', '--elec_ldf_path',
                        default=os.path.join(os.environ.get("PROJECT_DATA"),
                                             'trained_models',
                                             'temp'), help='')
    parser.add_argument('-cip', '--country_input_path',
                        default='', help='')
    parser.add_argument('-csp', '--country_shape_path',
                        default='', help='')
    parser.add_argument('-cop', '--country_output_path',
                        default='', help='')

    return parser.parse_args(args)


global_args = parse_args([])
global_args = global_args.__dict__

scaler_elec = pickle.load(open(global_args['elec_scaler_path'], 'rb'))
args_elec = json.load(open(os.path.join(global_args['elec_ldf_path'], 'args.json')))
data_keys_elec = json.load(open(global_args['elec_keys_path']))

def add_elec(tuple):
    (i,
     unique_file_id,
     country_input_path,
     country_shape_path,
     country_output_path,
     n_unique_file_ids,
     country_code) = tuple

    try:

        from ldf.models.model_ldf_scalar import LdfScalarModel
        from ldf.models.model_ldfbb import LdfBetaBinomModel

        model_elec = LdfBetaBinomModel(plotpostpred=args_elec['plotpostpred'],
                                       plotpost=args_elec['plotpost'],
                                       error_metric=args_elec['error_metric'],
                                       num_folds=args_elec['num_folds'],
                                       fold_num_test=args_elec['fold_num_test'],
                                       fold_num_val=args_elec['fold_num_val'],
                                       load_json=os.path.join(global_args['elec_ldf_path'], 'model.json'),
                                       load_weights=os.path.join(global_args['elec_ldf_path'], 'model.h5'),
                                       output_dir=r'/tmp',
                                       train_epochs=args_elec['train_epochs'],
                                       es_patience=args_elec['es_patience'],
                                       n_rand_restarts=args_elec['n_rand_restarts'],
                                       batch_size=args_elec['batch_size'],
                                       params=args_elec['params'],
                                       gpu=0,
                                       do_val=args_elec['do_val'],
                                       reg_mode=args_elec['reg_mode'],
                                       reg_val=args_elec['reg_val'],
                                       learning_rate_training=args_elec['learning_rate_training'],
                                       model_name=args_elec['model_name'],
                                       red_factor=args_elec['red_factor'],
                                       red_patience=args_elec['red_patience'],
                                       red_min_lr=args_elec['red_min_lr'],
                                       verbose=1)
        model_elec.fit(load_model=True,
                       save_model=False)

        density_geoms_geojson_path = os.path.join(country_input_path, f"{unique_file_id}_geoms.geojson")

        bldgs_gdf = gpd.read_file(density_geoms_geojson_path)
        bldgs_gdf["origin_origin_id"] = bldgs_gdf["origin"] + '_' + bldgs_gdf["origin_id"]

        print('loaded bldgs')

        ####################################
        # elec analysis
        ####################################

        # prepare data
        x_orig_elec = bldgs_gdf[data_keys_elec].values
        x_elec = scaler_elec.transform(x_orig_elec)

        mean_preds_elec, \
            preds_params_elec, \
            _ = \
            model_elec.predict(X=x_elec, n=np.ones(x_elec.shape[0]))

        # elec_mean_preds_params has priors and learned values separated.
        # Combining them here
        a_alls_elec = preds_params_elec['a_of_x'] + preds_params_elec['a0']
        b_alls_elec = preds_params_elec['b_of_x'] + preds_params_elec['b0']

        print(f'finished running elec for {unique_file_id}')

        ####################################
        # update and save buildings file
        ####################################

        bldgs_gdf['elec access (%)'] = mean_preds_elec * 100.
        bldgs_gdf['a_alls_elec'] = a_alls_elec
        bldgs_gdf['b_alls_elec'] = b_alls_elec

        try:
            bldgs_gdf.to_file(
                os.path.join(country_output_path, f"{unique_file_id}_geoms.geojson"),
                driver='GeoJSON')
            print(f'wrote files for grid cell: {unique_file_id}')

        except:
            print(f'Cannot write dataframe to file! It might be empty! Failed on grid cell: {unique_file_id}')
            with open(os.path.join(country_output_path, f"{unique_file_id}_geoms.err"), 'w') as f:
                f.write('Cannot write dataframe to file! It might be empty!')

    except:
        print(f'Something went wrong with {unique_file_id}. Debug for more info! \n')


def add_elec_by_grid_cell(country_input_path,
                             country_shape_path,
                             country_output_path,
                             par_pool_size,
                             country_code):
    all_files = np.array([file for file in os.listdir(country_input_path) if file.endswith("_geoms.geojson")])
    # density_geoms.geojson
    all_files_underscore_index = np.char.find(all_files, '_')
    all_files_ids = np.char.ljust(all_files, all_files_underscore_index)
    unique_file_ids = np.unique(all_files_ids)
    n_unique_file_ids = unique_file_ids.size

    par_list = []
    for i, unique_file_id in enumerate(unique_file_ids):

        elec_geoms_csv_path = os.path.join(country_output_path, f"{unique_file_id}_geoms.csv")
        elec_geoms_geojson_path = os.path.join(country_output_path, f"{unique_file_id}_geoms.geojson")

        # if the files already exist, skip it. i.e. only run if there's a missing file
        if not (os.path.exists(elec_geoms_csv_path) and os.path.exists(elec_geoms_geojson_path)):
            print(f'added {unique_file_id} to the list for generating density geometries.')
            par_list.append((i,
                             unique_file_id,
                             country_input_path,
                             country_shape_path,
                             country_output_path,
                             n_unique_file_ids,
                             country_code))

    with multiprocessing.Pool(par_pool_size) as p:
        print(p.map(add_elec, par_list, chunksize=1))
    # add_elec(par_list[0])


###################################################
def main(args):
    args = parse_args(args)
    args = args.__dict__

    os.makedirs(args['country_output_path'], exist_ok=True)

    print(f"running add_elec_by_grid_cell")
    add_elec_by_grid_cell(args['country_input_path'],
                             args['country_shape_path'],
                             args['country_output_path'],
                             args['par_pool_size'],
                             args['country_code'])

    print('done!')


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
