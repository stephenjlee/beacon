import os, sys, argparse, json, time, pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.preprocessing import StandardScaler

from pathlib import Path

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
data_path = Path(os.environ.get("PROJECT_DATA"))
sys.path.append(os.environ.get("PROJECT_ROOT"))
sys.path.append(os.environ.get("LDF_ROOT"))

from beacon.visualization.geoprocessing import save_geojson

from ldf.models.model_ldfbb import LdfBetaBinomModel
import ldf.utils.utils_general as ug


def parse_args(args):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Run sampler to infer electrification status.')
    parser.add_argument('-bp', '--bldgs_path',
                        default='',
                        help='')
    parser.add_argument('-r', '--rows', type=int, default=None,
                        help='the number of rows to load')
    parser.add_argument('-s', '--rng_seed', type=int, default=0,
                        help='the integer seed for the random number generator')
    parser.add_argument('-v', '--verbose', type=int, default=1,
                        help='verbosity for each Keras nn training epoch, {0, 1, 2} [default 0]')
    parser.add_argument('-sgj', '--save_geojson', default=False,
                        help='Whether (if True) to save a geojson, or (if False) just a csv. Saving a geojson requires significantly more memory.')
    parser.add_argument('-tr', '--train_epochs', type=int, default=2000,
                        help='the number epochs to train the nn on')
    parser.add_argument('-m', '--model_name', default='LARGE',
                        help='the name of the model to run. Current options include: SHALLOW, LARGER, XLARGE')
    parser.add_argument('-pms', '--params', default='{"a0": 1.0, "b0": 1.0, "nu": 1.0}',
                        help='the param settings to use')
    parser.add_argument('-nf', '--num_folds', type=int, default=0,
                        help='the number of folds for k-fold cross-validation')
    parser.add_argument('-fnt', '--fold_num_test', type=int, default=0,
                        help='the fold number for k-fold cross-validation')
    parser.add_argument('-fnv', '--fold_num_val', type=int, default=0,
                        help='the fold number for k-fold cross-validation')
    parser.add_argument('-lm', '--load_model', type=ug.str2bool, nargs='?',
                        const=True, default=False, help='whether to load a saved model, or run from scratch.')
    parser.add_argument('-lj', '--load_json',
                        default=f'',
                        help='if load_model is True: model file to load')
    parser.add_argument('-lw', '--load_weights',
                        default=f'',
                        help='if load_weights is True: weights file to load. Note that this model needs to be '
                             'compatible with the rest of the args given.')
    parser.add_argument('-g', '--gpu', type=int, default=0,
                        help='1: use the GPU; 0: just use CPUs')
    parser.add_argument('-bs', '--batch_size', type=int, default=512,
                        help='batch size for stochastic gradient descent')
    parser.add_argument('-ep', '--es_patience', type=int, default=50,
                        help='early stopping patience, default -1 (no early stopping)')
    parser.add_argument('-nrr', '--n_rand_restarts', type=int, default=1,
                        help='number of random restarts for model training')
    parser.add_argument('-do', '--do_val', type=float, default=0.05,
                        help='the dropout value to employ at the hidden layers')
    parser.add_argument('-rm', '--reg_mode', default='L2',
                        help='the regularization mode to employ. Available options are L2 and L1')
    parser.add_argument('-rv', '--reg_val', type=float, default=0.00,
                        help='the regularization value to employ with the regularization mode specified by --reg_mode')
    parser.add_argument('-ppp', '--plotpostpred', type=bool, default=True,
                        help='whether or not to calculate and plot posterior predictive metrics')
    parser.add_argument('-pp', '--plotpost', type=bool, default=True,
                        help='whether or not to calculate and plot posterior metrics')
    parser.add_argument('-em', '--error_metric', default='betabinom',
                        help='the name of the eror metric set to use. Current options include: betabinom')
    parser.add_argument('-od', '--output_dir',
                        default=os.path.join(os.environ.get("PROJECT_ROOT"), 'out', 'beacon'),
                        help='the output directory')
    parser.add_argument('-lrt', '--learning_rate_training', type=float, default=0.001,
                        help='the learning rate to apply during training')
    parser.add_argument('-rflr', '--red_factor', type=float, default=0.5,
                        help='ReduceLROnPlateau factor')
    parser.add_argument('-rplr', '--red_patience', type=float, default=50,
                        help='ReduceLROnPlateau patience')
    parser.add_argument('-rmlr', '--red_min_lr', type=float, default=0.00000001,
                        help='ReduceLROnPlateau minimum learning rate')

    return parser.parse_args(args)


def load_bldgs(bldgs_path, rows):
    gdf = gpd.read_file(bldgs_path, rows=rows)

    return gdf


def load_data(bldgs_path,
              rows):
    bldgs_df_all = load_bldgs(bldgs_path, rows)

    # Step 1: Define the grid
    minx, miny, maxx, maxy = bldgs_df_all.total_bounds
    grid_size = 0.025

    # Generate the x and y boundaries of the grid
    x_grid = list(np.arange(minx, maxx, grid_size))
    y_grid = list(np.arange(miny, maxy, grid_size))

    # Step 2: Assign each building to a grid cell
    bldgs_df_all['x_grid'] = pd.cut(bldgs_df_all.centroid.geometry.x, bins=x_grid, labels=False, right=False)
    bldgs_df_all['y_grid'] = pd.cut(bldgs_df_all.centroid.geometry.y, bins=y_grid, labels=False, right=False)

    # Create a unique grid cell identifier
    bldgs_df_all['grid_cell'] = bldgs_df_all['x_grid'].astype(str) + "_" + bldgs_df_all['y_grid'].astype(str)

    # Step 3: Sort the grid cells by the number of buildings
    sorted_cells = bldgs_df_all.groupby('grid_cell').size().sort_values(ascending=False).index.tolist()

    # Step 4: Assign grid cells to train, validation, and test sets in an alternating manner
    train_cells = []
    val_cells = []
    test_cells = []

    for i, cell in enumerate(sorted_cells):
        if i % 3 == 0:
            train_cells.append(cell)
        elif i % 3 == 1:
            val_cells.append(cell)
        else:
            test_cells.append(cell)

    bldgs_df_train = bldgs_df_all[bldgs_df_all['grid_cell'].isin(train_cells)]
    bldgs_df_val = bldgs_df_all[bldgs_df_all['grid_cell'].isin(val_cells)]
    bldgs_df_test = bldgs_df_all[bldgs_df_all['grid_cell'].isin(test_cells)]

    # Calculate the centroids
    bldgs_df_all['centroid'] = bldgs_df_all.geometry.centroid

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot centroids for each dataset with different colors
    bldgs_df_all.loc[bldgs_df_all['grid_cell'].isin(train_cells), 'centroid'].plot(ax=ax, color='red', markersize=1,
                                                                                   label='Train')
    bldgs_df_all.loc[bldgs_df_all['grid_cell'].isin(val_cells), 'centroid'].plot(ax=ax, color='green', markersize=1,
                                                                                 label='Validation')
    bldgs_df_all.loc[bldgs_df_all['grid_cell'].isin(test_cells), 'centroid'].plot(ax=ax, color='blue', markersize=1,
                                                                                  label='Test')

    # Legend
    ax.legend()

    # Title and display
    ax.set_title('Distribution of data across train, validation, and test sets')
    plt.savefig('distribution.png')
    plt.close()

    keys = ["area_in_meters",
            "n_bldgs_1km_away",
            "lulc2017_built_area_N1",
            "lulc2017_rangeland_N1",
            "lulc2017_crops_N1",
            "lulc2017_built_area_N11",
            "lulc2017_rangeland_N11",
            "lulc2017_crops_N11",
            "ntl2018_N1",
            "ntl2018_N11",
            "ookla_fixed_20200101_avg_d_kbps",
            "ookla_fixed_20200101_devices",
            "ookla_mobile_20200101_avg_d_kbps",
            "ookla_mobile_20200101_devices"]

    scaler = StandardScaler()
    scaler.fit(bldgs_df_all[keys])

    x = scaler.transform(bldgs_df_all[keys])
    x_train = scaler.transform(bldgs_df_train[keys])
    x_val = scaler.transform(bldgs_df_val[keys])
    x_test = scaler.transform(bldgs_df_test[keys])

    y = bldgs_df_all['electrified_based_on_rule'].values.astype(float)
    y_train = bldgs_df_train['electrified_based_on_rule'].values.astype(float)
    y_val = bldgs_df_val['electrified_based_on_rule'].values.astype(float)
    y_test = bldgs_df_test['electrified_based_on_rule'].values.astype(float)

    ids = bldgs_df_all['origin_origin_id'].values
    ids_train = bldgs_df_test['origin_origin_id'].values
    ids_val = bldgs_df_test['origin_origin_id'].values
    ids_test = bldgs_df_test['origin_origin_id'].values

    n = np.ones(x.shape[0])
    n_train = np.ones(x_train.shape[0])
    n_val = np.ones(x_val.shape[0])
    n_test = np.ones(x_test.shape[0])

    data = {
        'x': x,
        'y': y,
        'n': n,
        'ids': ids,
        'x_train': x_train,
        'x_val': x_val,
        'x_test': x_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'n_train': n_train,
        'n_val': n_val,
        'n_test': n_test,
        'ids_train': ids_train,
        'ids_val': ids_val,
        'ids_test': ids_test
    }

    return data, bldgs_df_all, scaler, keys


def run_elec_ldf(args):
    start_time = time.time()
    (data,
     bldgs_df,
     scaler,
     keys) = load_data(args['bldgs_path'],
                       args['rows'])

    print(f'Loading data: {(time.time() - start_time) / 60:0.1f} minutes')

    #############################################
    # initialize and fit model
    #############################################
    start_time = time.time()
    model = LdfBetaBinomModel(plotpostpred=args['plotpostpred'],
                              plotpost=args['plotpost'],
                              error_metric=args['error_metric'],
                              num_folds=args['num_folds'],
                              fold_num_test=args['fold_num_test'],
                              fold_num_val=args['fold_num_val'],
                              load_json=args['load_json'],
                              load_weights=args['load_weights'],
                              output_dir=args['output_dir'],
                              train_epochs=args['train_epochs'],
                              es_patience=args['es_patience'],
                              n_rand_restarts=args['n_rand_restarts'],
                              batch_size=args['batch_size'],
                              params=args['params'],
                              gpu=args['gpu'],
                              do_val=args['do_val'],
                              reg_mode=args['reg_mode'],
                              reg_val=args['reg_val'],
                              learning_rate_training=args['learning_rate_training'],
                              model_name=args['model_name'],
                              red_factor=args['red_factor'],
                              red_patience=args['red_patience'],
                              red_min_lr=args['red_min_lr'],
                              verbose=1
                              )
    print(f'Instantiating model: {(time.time() - start_time) / 60:0.1f} minutes')

    start_time = time.time()
    model.fit(X=data['x_train'],
              y=data['y_train'],
              n=data['n_train'],
              X_val=data['x_val'],
              y_val=data['y_val'],
              n_val=data['n_val'],
              load_model=args['load_model'])
    print(f'Fitting model: {(time.time() - start_time) / 60:0.1f} minutes')

    start_time = time.time()
    model.predict_for_train(data['x_train'],
                            data['y_train'],
                            data['n_train'])
    model.predict_for_val(data['x_val'],
                          data['y_val'],
                          data['n_val'])
    model.predict_for_test(data['x_test'],
                           data['y_test'],
                           data['n_test'])
    print(f'Predicting model: {(time.time() - start_time) / 60:0.1f} minutes')

    start_time = time.time()
    model.plot_postpred(data['ids_test'],
                        data['y_train'],
                        data['y_test'],
                        data['n_test'])

    model.plot_posterior(data['y_test'],
                         data['n_test'])

    model.plot_history()
    model.plot_calibration(data['y_train'],
                           data['y_val'],
                           data['y_test'])
    print(f'Plotting model: {(time.time() - start_time) / 60:0.1f} minutes')

    start_time = time.time()

    # save scaler and data keys
    pickle.dump(scaler, open(os.path.join(model.get_output_dir(), 'scaler.pkl'), 'wb'))
    with open(os.path.join(model.get_output_dir(), 'data_keys.json'), "w") as outfile:
        json.dump(keys, outfile)

    # save args
    with open(os.path.join(model.get_output_dir(), 'args.json'), "w") as outfile:
        json.dump(args, outfile)

    model.save_summary(data['x_train'],
                       data['x_val'],
                       data['x_test'],
                       data['y_train'],
                       data['y_val'],
                       data['y_test'],
                       data['n_train'],
                       data['n_val'],
                       data['n_test'],
                       data['ids_train'],
                       data['ids_val'],
                       data['ids_test'])
    print(f'Run summary: {(time.time() - start_time) / 60:0.1f} minutes')

    mean_preds_all, \
        preds_params_all, \
        preds_params_flat_all = model.predict(X=data['x'],
                                              n=data['n'])

    # save geojson
    if args['save_geojson']:
        save_geojson(bldgs_df,
                     data,
                     preds_params_all,
                     os.path.join(model.get_output_dir(), 'web', 'data'))  # todo: uncomment
        print('finished saving geojson and copying web project')


def run_with_args_subset(args_subset):
    args = parse_args([])
    args = args.__dict__

    print('asdf')

    for args_elem in args_subset.keys():
        args[args_elem] = args_subset[args_elem]

    run_elec_ldf(args)


def main(args):
    args = parse_args(args)
    args = args.__dict__

    run_elec_ldf(args)


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
