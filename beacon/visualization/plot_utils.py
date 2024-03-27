import json, os
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def plot_intensity_vs_density(density_cells, intensity_cells, title='Title', xlabel='X Label', ylabel='Y Label'):
    # plot light intensity vs building density
    area = 10
    fig, ax = plt.subplots()
    cax = ax.scatter(density_cells.flatten(), intensity_cells.flatten(), s=area, alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    return fig, cax, ax


def plot_heatmap(density_cells, title='Title', log_scale=False, cmap='hot', xlabel='', ylabel='', vmin=None, vmax=None, aspect=None, fname=None):
    fig, ax = plt.subplots()
    ax.set_title(title)
    if xlabel != '':
        ax.set_xlabel(xlabel)
    if ylabel != '':
        ax.set_ylabel(ylabel)
    if log_scale == False:
        cax = ax.imshow(density_cells, cmap=cmap, interpolation='nearest', origin='lower', vmin=vmin, vmax=vmax, aspect=aspect)
    else:
        cax = ax.imshow(density_cells + 1,
                        norm=colors.LogNorm(vmin=np.nanmin(density_cells + 1.0), vmax=np.nanmax(density_cells + 1.0)),
                        cmap=cmap, interpolation='nearest', origin='lower', aspect=aspect)
    cbar = fig.colorbar(cax)
    if fname is not None:
        plt.savefig(fname)

    return fig, cax, ax, cbar


def plot_lineplot(x, y, title='Title', xlabel='', ylabel='', fname=None):
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_title(title)
    if xlabel != '':
        ax.set_xlabel(xlabel)
    if ylabel != '':
        ax.set_ylabel(ylabel)
    if fname is not None:
        plt.savefig(fname)

    return fig, ax


def plot_phi(phi, title='', fname=None):
    fig, ax = plt.subplots()
    xv = np.linspace(0,1,1000)
    av = phi[0]*xv + phi[1]
    bv = phi[2]*xv + phi[3]
    ax.plot(xv, av, label='num heads')
    ax.plot(xv, bv, label='num tails')
    ax.plot(xv, av+bv, label='num trials')
    ax.set_title(title)
    ax.grid()
    ax.legend()
    if fname is not None:
        plt.savefig(fname)

    return fig, ax


def animate_thetas(all_theta, Y, video_name=None):
    im_shape = (Y['c']['columns'], Y['c']['rows'])

    fig = plt.figure()
    data = np.flipud(all_theta[:, 0].reshape(im_shape, order='F'))

    ax = fig.add_subplot(111)
    ax.set_title("Iter {}".format(1))
    ax.set_aspect('equal')

    im = ax.imshow(data, cmap='hot', vmin=0, vmax=1, interpolation='nearest')

    im.set_clim([0, 1])

    def update_img(n):
        im.set_data(np.flipud(all_theta[:, n].reshape(im_shape, order='F')))
        ax.set_title("Iter {}".format(n+1))
        return im

    ani = animation.FuncAnimation(
        fig, update_img, all_theta.shape[1], interval=25, repeat=False)

    if video_name is not None:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)
        ani.save(video_name, writer=writer)

def save_heatmap_data(cell_data, mean_data, std_data):
    mean_outer_list = []
    std_outer_list = []
    for i in range(cell_data['columns']):
        for j in range(cell_data['rows']):
            if not np.isnan(mean_data[i, j]):
                mean_box_dict = {
                    'type': 'Feature',
                    'properties': {"color": mean_data[i][j] / np.nanmax(mean_data)},
                    'geometry': {
                        'type': 'Polygon',
                        'coordinates': [[
                            [cell_data['geom'][i, j]['left'],
                             cell_data['geom'][i, j]['top']],
                            [cell_data['geom'][i, j]['right'],
                             cell_data['geom'][i, j]['top']],
                            [cell_data['geom'][i, j]['right'],
                             cell_data['geom'][i, j]['bottom']],
                            [cell_data['geom'][i, j]['left'],
                             cell_data['geom'][i, j]['bottom']]
                        ]]
                    }
                }
                std_box_dict = {
                    'type': 'Feature',
                    'properties': {"color": std_data[i][j] / np.nanmax(std_data)},
                    'geometry': {
                        'type': 'Polygon',
                        'coordinates': [[
                            [cell_data['geom'][i, j]['left'],
                             cell_data['geom'][i, j]['top']],
                            [cell_data['geom'][i, j]['right'],
                             cell_data['geom'][i, j]['top']],
                            [cell_data['geom'][i, j]['right'],
                             cell_data['geom'][i, j]['bottom']],
                            [cell_data['geom'][i, j]['left'],
                             cell_data['geom'][i, j]['bottom']]
                        ]]
                    }
                }
                mean_outer_list.append(mean_box_dict)
                std_outer_list.append(std_box_dict)
    mean_dump_object = {
        "type": "FeatureCollection",
        "features": mean_outer_list
    }
    std_dump_object = {
        "type": "FeatureCollection",
        "features": std_outer_list
    }
    with open(r'web/mean_data.js', 'w') as outfile:
        outfile.write('var data = ')
        json.dump(mean_dump_object, outfile)
    with open(r'web/std_data.js', 'w') as outfile:
        outfile.write('var data = ')
        json.dump(std_dump_object, outfile)



def save_heatmap_data_with_attrs(cell_data, mean_data, std_data, run_name_path, label):
    mean_outer_list = []
    std_outer_list = []
    min_mean = np.nanmin(mean_data)
    max_mean = np.nanmax(mean_data)
    min_std = np.nanmin(std_data)
    max_std = np.nanmax(std_data)
    for i in range(cell_data['columns']):
        for j in range(cell_data['rows']):
            if not np.isnan(mean_data[i, j]):

                mean_box_dict = {
                    'type': 'Feature',
                    'properties': {
                        "color": (mean_data[i][j] - min_mean) / (np.nanmax(mean_data) - min_mean),
                        "val": mean_data[i][j]
                                   },
                    'geometry': {
                        'type': 'Polygon',
                        'coordinates': [[
                            [cell_data['geom'][i, j]['left'],
                             cell_data['geom'][i, j]['top']],
                            [cell_data['geom'][i, j]['right'],
                             cell_data['geom'][i, j]['top']],
                            [cell_data['geom'][i, j]['right'],
                             cell_data['geom'][i, j]['bottom']],
                            [cell_data['geom'][i, j]['left'],
                             cell_data['geom'][i, j]['bottom']]
                        ]]
                    }
                }
                std_box_dict = {
                    'type': 'Feature',
                    'properties': {
                        "color": (std_data[i][j] - min_std)/(np.nanmax(std_data) - min_std),
                        "val": std_data[i][j]
                                   },
                    'geometry': {
                        'type': 'Polygon',
                        'coordinates': [[
                            [cell_data['geom'][i, j]['left'],
                             cell_data['geom'][i, j]['top']],
                            [cell_data['geom'][i, j]['right'],
                             cell_data['geom'][i, j]['top']],
                            [cell_data['geom'][i, j]['right'],
                             cell_data['geom'][i, j]['bottom']],
                            [cell_data['geom'][i, j]['left'],
                             cell_data['geom'][i, j]['bottom']]
                        ]]
                    }
                }
                mean_outer_list.append(mean_box_dict)
                std_outer_list.append(std_box_dict)
    mean_dump_object = {
        "type": "FeatureCollection",
        "features": mean_outer_list
    }
    std_dump_object = {
        "type": "FeatureCollection",
        "features": std_outer_list
    }
    with open(os.path.join(run_name_path, 'mean_data_fold_' + str(label))+'.js', 'w') as outfile:
        outfile.write('var data = ')
        json.dump(mean_dump_object, outfile)
    with open(os.path.join(run_name_path, 'std_data_fold_' + str(label))+'.js', 'w') as outfile:
        outfile.write('var data = ')
        json.dump(std_dump_object, outfile)