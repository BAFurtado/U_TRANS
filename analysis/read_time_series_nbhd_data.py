import os
import sys

import pickle

import matplotlib.pyplot as plt
import pandas as pd


if os.path.abspath('.') not in sys.path:
    sys.path.append(os.path.abspath('.'))

import read_data

group_list = [['center']]
for ring in '123':
    group_list.append(['ring' + ring + i for i in 'abc'])
    group_list.append(['ring' + ring + i for i in 'def'])
group_names = ['center', '1-old', '1-new', '2-old', '2-new', '3-old', '3-new']
groups = dict()
for i, name in enumerate(group_names):
    groups[name] = group_list[i]


def read_data2(path, hoods):
    # 1. Getting files
    # 2. Returns a dictionary of scenarios containing files
    data, results = dict(), dict()
    for scene in os.listdir(path):
        if scene.isupper():
            temp_path = os.path.join(path, scene, 'neighbourhoods')
            data[scene] = os.listdir(temp_path)
            results[scene] = dict()
            for hood in hoods:
                full = pd.DataFrame()
                for d in data[scene]:
                    if hood in d:
                        temp = pd.read_csv(os.path.join(temp_path, d))
                        temp = temp.reset_index()
                        full = full.append(temp)
                results[scene][hood] = full.groupby('index').agg('mean').reset_index()
    return results


def aggregate_nbhd_data(scenarios_data):
    """ Receives the dictionary of scenarios containing individual neighbourhood data.
    """
    output = dict()
    for scenario in scenarios_data:
        output[scenario] = dict()
        for group in groups:
            output[scenario][group] = pd.DataFrame()
            for nbhd in groups[group]:
                output[scenario][group] = output[scenario][group].append(scenarios_data[scenario][nbhd])
            output[scenario][group] = output[scenario][group].groupby('index').agg('mean')
    return output


def get_aggregated_shp(shp_name='hexagons'):
    s = read_data.read_shapes(shp_name)
    for key in groups:
        s.loc[s['Name'].apply(lambda x: x in groups[key]), 'group'] = key
    s = s.dissolve(by='group').reset_index()
    return s[['group', 'geometry']]


def plot_nbhd_series_scenarios(data, path):
    for col in data['BASELINE']['center'].columns:
        fig, axes = plt.subplots(2, 2)
        plt.set_cmap('Spectral')
        axes = axes.flat
        fig.suptitle(col, fontsize=16)
        for i, key in enumerate(data):
            for nbhd in data[key]:
                data[key][nbhd][col].plot(ax=axes[i], label=nbhd, lw=.75)
            axes[i].set_title(key)
            [axes[i].spines[j].set_visible(False) for j in ['top', 'bottom', 'right', 'left']]
        axes[0].legend(frameon=False,
                       loc='lower left',
                       bbox_to_anchor=(1.06, 1.01),
                       mode='expand',
                       ncol=2)
        fig.tight_layout()
        plt.savefig(os.path.join(path, f'{col}.png'))
        plt.show()


if __name__ == '__main__':
    if os.path.split(os.getcwd())[1] == 'analysis':
        os.chdir(os.path.split(os.getcwd())[0])

    if not os.path.exists('output/nbhd_data'):
        nbhds = ['center']
        for t in '123':
            for z in 'abcdef':
                nbhds.append(f'ring{t}{z}')
        p = 'output'
        res = read_data2(p, nbhds)
        with open('output/nbhd_data', 'wb') as handler:
            pickle.dump(res, handler)
    else:
        with open('output/nbhd_data', 'rb') as handler:
            res = pickle.load(handler)
    out = aggregate_nbhd_data(res)
    # agg_shape = get_aggregated_shp()
    # agg_shape.to_file('output/aggregated_shp.shp')
    p = 'output/scenarios_plots'
    plot_nbhd_series_scenarios(out, p)

