import os, sys, argparse
import geopandas as gpd
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
sys.path.append(os.environ.get("PROJECT_ROOT"))


def parse_args(args):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='')

    parser.add_argument('-bp', '--bldgs_path',
                        default=os.path.join(os.environ.get("PROJECT_OUT"),
                                             'RWA_elec_buildings_sjoin.geojson'),
                        help='')
    parser.add_argument('-sgj', '--save_geojson', default=False,
                        help='Whether (if True) to save a geojson, or (if False) just a csv. '
                             'Saving a geojson requires significantly more memory.')
    parser.add_argument('-r', '--rows', type=int, default=20000, help='the number of rows to load')

    return parser.parse_args(args)


def load_data(bldgs_path, rows, save_geojson):
    if save_geojson:
        bldgs_df = gpd.read_file(bldgs_path, rows=rows)
    else:
        bldgs_df = gpd.read_file(bldgs_path, rows=rows, ignore_geometry=True)

    return bldgs_df


def main(args):
    args = parse_args(args)
    args = args.__dict__

    bldgs_df = load_data(args['bldgs_path'], args['rows'], args['save_geojson'])

    print(bldgs_df)

    print('done!')


if __name__ == "__main__":
    main(sys.argv[1:])
