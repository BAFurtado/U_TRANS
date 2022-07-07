import os

import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd


def read_data(path):
    # 1. Getting SHP files
    # 2. Returns a dictionary of scenarios containing SHP files names
    data = dict()
    for d in os.listdir(path):
        if d.isupper():
            temp_path = os.path.join(path, d)
            data[d] = os.listdir(temp_path)
            data[d] = [gpd.read_file(os.path.join(temp_path, f)) for f in data[d] if f.endswith('.shp')]
    return data


def averaging_data(data):
    data_df = dict()
    for key in data:
        data_df[key] = pd.DataFrame()
        for each in data[key]:
            data_df[key] = data_df[key].append(each)
        geo = data_df[key][['Name', 'geometry']].drop_duplicates()
        data_df[key] = data_df[key].groupby('Name').agg('mean').reset_index()
        data_df[key] = data_df[key].merge(geo, on='Name')
    return data_df


def plotting(data, path):
    for col in data['BASELINE'].columns:
        if col not in ['Name', 'geometry']:
            fig, axes = plt.subplots(2, 2)
            axes = axes.flat
            fig.suptitle(col, fontsize=16)
            for i, key in enumerate(data):
                data[key].plot(column=col, ax=axes[i])
                axes[i].set_title(key)
                axes[i].spines['top'].set_visible(False)
                axes[i].spines['bottom'].set_visible(False)
                axes[i].spines['right'].set_visible(False)
                axes[i].spines['left'].set_visible(False)
            fig.tight_layout()
            plt.savefig(os.path.join(path, f'{col}.png'))
            plt.show()


if __name__ == '__main__':
    p = '../output'
    dta = read_data(p)
    dta = averaging_data(dta)
    plotting(dta, p)

