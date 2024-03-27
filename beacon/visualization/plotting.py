import sys, os
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('bmh')

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
sys.path.append(os.environ.get("PROJECT_ROOT"))

def plot_building_types_vs_a_b(bldgs_df, metrics_df, output_folder_name):


    # plotting a and b vs building type
    import seaborn as sns
    import matplotlib.pyplot as plt
    n_flds = metrics_df['fold_num'].unique().size - 1
    for fld in range(n_flds):
        a_of_x_train_fld = metrics_df[(metrics_df['metric'] == 'a_of_x_train')]['val'].values[fld]
        b_of_x_train_fld = metrics_df[(metrics_df['metric'] == 'b_of_x_train')]['val'].values[fld]
        alpha_train_fld = metrics_df[(metrics_df['metric'] == 'alpha_train')]['val'].values[fld]
        beta_train_fld = metrics_df[(metrics_df['metric'] == 'beta_train')]['val'].values[fld]
        a_train = a_of_x_train_fld + alpha_train_fld
        b_train = b_of_x_train_fld + beta_train_fld

        a_of_x_test_fld = metrics_df[(metrics_df['metric'] == 'a_of_x_test')]['val'].values[fld]
        b_of_x_test_fld = metrics_df[(metrics_df['metric'] == 'b_of_x_test')]['val'].values[fld]
        alpha_test_fld = metrics_df[(metrics_df['metric'] == 'alpha_test')]['val'].values[fld]
        beta_test_fld = metrics_df[(metrics_df['metric'] == 'beta_test')]['val'].values[fld]
        a_test = a_of_x_test_fld + alpha_test_fld
        b_test = b_of_x_test_fld + beta_test_fld

        ids_train_fld = metrics_df[(metrics_df['metric'] == 'ids_train')]['val'].values[fld]
        ids_test_fld = metrics_df[(metrics_df['metric'] == 'ids_test')]['val'].values[fld]

        train_df = pd.DataFrame.from_dict({
            'a': a_train,
            'b': b_train,
            'origin_origin_id': ids_train_fld
        })
        test_df = pd.DataFrame.from_dict({
            'a': a_test,
            'b': b_test,
            'origin_origin_id': ids_test_fld
        })


        train_df = train_df.merge(bldgs_df[['origin_origin_id', 'building_types']],
                       left_on='origin_origin_id',
                       right_on='origin_origin_id')

        fig, axs = plt.subplots(3, figsize=(8, 12), sharex=True, sharey=True)

        sns.scatterplot(data=train_df[~train_df['building_types'].isnull()],
                        x="a",
                        y="b",
                        hue="building_types",
                        palette="deep",
                        ax=axs[0])
        axs[0].set(xlabel='$\\alpha$',
                   ylabel='$\\beta$')
        axs[0].set_title(f'Building Type vs $\\alpha$ & $\\beta$, Fold {fld}')

        sns.scatterplot(data=train_df[train_df['building_types'] == 'Residential'],
                        x="a",
                        y="b",
                        color='#4C72B0',
                        ax=axs[1])
        axs[1].set(xlabel='$\\alpha$',
                   ylabel='$\\beta$')
        axs[1].set_title(f'Residential Only')

        sns.scatterplot(data=train_df[train_df['building_types'] == 'Non Residential'],
                        x="a",
                        y="b",
                        color='#DD8452',
                        ax=axs[2])
        axs[2].set(xlabel='$\\alpha$',
                   ylabel='$\\beta$')
        axs[2].set_title(f'Non-Residential Only')

        plt.tight_layout()
        output_path = os.path.join(output_folder_name, f'bldg_vs_a_b_fld_{fld}.pdf')
        plt.savefig(output_path)
        plt.close('all')

