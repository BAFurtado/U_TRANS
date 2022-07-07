import os

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))


def finish_plot(ax, col, path):
    plt.rcParams.update({'font.size': 9})
    ax.legend(frameon=False)
    ax.set(xlabel='Time', ylabel=col, title=f'{col}')
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.yaxis.set_major_formatter(plt.FuncFormatter('{:.2f}'.format))
    plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)
    plt.tick_params(axis='both', which='both', bottom=False, top=False,
                    labelbottom=True, left=False, right=False, labelleft=True)
    fmt = 'png'
    p = os.path.join(path, fr'{col}.{fmt}')
    plt.savefig(p, bbox_inches='tight')
    # plt.show()


def plotting(data, path, to_eliminate=0):
    if to_eliminate > 0:
        # data = data.loc[len(data['current_step']) * to_eliminate:, :]
        data = data.loc[to_eliminate:, :]
    for col in data.columns:
        if col != 'current_step':
            fig, ax = plt.subplots()
            ax.plot(data.current_step, data[col], label=col)
            finish_plot(ax, col, path)
            plt.close()


def plot_neighbourhoods(data, path, run_id, to_eliminate=0):
    """ Data is a dictionary with neighbourhood names as keys and neighbourhood data as a DataFrame
    """
    keys = list(data.keys())
    if to_eliminate > 0:
        for key in keys:
            data[key] = data[key].loc[to_eliminate:, :]
    for col in data[keys[0]].columns:
        fig, ax = plt.subplots()
        for key in keys:
            ax.plot(data[key][col], label=key)
        finish_plot(ax, col, path, run_id)
        plt.close()


def plot_geo(data, path, run_id, col, to_eliminate=0):
    """Generate a spatial plot for data at the end of the simulation"""
#    if to_eliminate > 0:
#        data = data.loc[len(data['current_step']) * to_eliminate:, :]
    cmap = cm.get_cmap('seismic')
    fig, ax = plt.subplots(figsize=(15, 15), subplot_kw={'aspect': 'equal'})

    # Background
    for name in data['Name']:
        shape_select = data[data['Name'] == str(name)]
        shape_select.plot(ax=ax, color='grey', linewidth=0.6, alpha=.7, edgecolor='black')

    # Data plot
    ax = data.plot(column=col, cmap=cmap, alpha=.5, ax=ax)
    data.apply(lambda x: ax.annotate(text=x['Name'].replace("ring", ""), xy=x.geometry.centroid.coords[0], ha='center',
                                     size=25), axis=1)
    cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
    sm = plt.cm.ScalarMappable(cmap='seismic', norm=plt.Normalize(vmin=0, vmax=1))
    sm._A = []
    fig.colorbar(sm, cax=cax)

    # Adding the grid location, title, axes labels
    ax.grid(True, color='grey', linestyle='-')
    ax.set_title(col.capitalize().replace('_', ' '))
    ax.set_xlabel('Longitude (in degrees)')
    ax.set_ylabel('Latitude (in degrees)')
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    fmt = 'png'
    p = os.path.join(path, fr'geo_{run_id}_{col}.{fmt}')
    plt.savefig(p, bbox_inches='tight')


def plot_geo_descript(shape_name, col_name, map_name, path, time_id, run_id):
    # Data plot
    dta = gpd.read_file(f'data/{shape_name}.shp')
    dta['coords'] = dta['geometry'].apply(lambda x: x.representative_point().coords[:])
    dta['coords'] = [coords[0] for coords in dta['coords']]
    f, ax = plt.subplots(1)
    dta.plot(ax=ax)
    for idx, row in dta.iterrows():
        plt.annotate(text=row[col_name], xy=row['coords'],
                     horizontalalignment='center', size=25)
    f.suptitle(map_name)
    fmt = 'png'
    p = os.path.join(path, fr'geo_{time_id}_{run_id}_{map_name}.{fmt}')
    plt.savefig(p, bbox_inches='tight')


def plot_geo_business(background, businesses, path, time_id, run_id):
    fig, ax = plt.subplots(figsize=(15, 15), subplot_kw={'aspect': 'equal'})
    # Background
    for name in background['Name']:
        shape_select = background[background['Name'] == str(name)]
        shape_select.plot(ax=ax, color='moccasin', linewidth=0.6, alpha=.7, edgecolor='black')
    dta = background.copy(deep=True)
    dta['coords'] = dta['geometry'].apply(lambda x: x.representative_point().coords[:])
    dta['coords'] = [coords[0] for coords in dta['coords']]
    text_scale = 25
    dot_scale = 5
    for idx, row in dta.iterrows():
        plt.annotate(text=row['Name'].replace("ring", ""), xy=row['coords'],
                     horizontalalignment='center', size=text_scale)
    labels = {'old': 'tab:blue', 'new': 'tab:green', 'service': 'tab:orange', }
    for industry in labels:
        xs = [point.location.x for point in businesses if point.industry.name == industry]
        ys = [point.location.y for point in businesses if point.industry.name == industry]
        sizes = [dot_scale * b.target_size for b in businesses if b.industry.name == industry]
        ax.scatter(xs, ys, s=sizes, c=labels[industry], label=industry, alpha=0.8, edgecolors='none')
    ax.legend(frameon=False, labelspacing=2, fontsize='xx-large')  # 'xx-large'
    ax.grid(True)
    ax.set_title('Industries location', fontsize=text_scale)
    ax.set_xlabel('Longitude (in degrees)', fontsize=text_scale - 2)
    ax.set_ylabel('Latitude (in degrees)', fontsize=text_scale - 2)
    fmt = 'png'
    p = os.path.join(path, fr'geo_{time_id}_{run_id}_business.{fmt}')
    plt.savefig(p, bbox_inches='tight')
    # plt.show()
