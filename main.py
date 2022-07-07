import collections.abc
import copy
import os
import sys
import time
from collections import defaultdict

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from joblib import Parallel, delayed

import params
import scenarios as scenes
import simulation
from analysis import plotting

color_map = [['blue', 'blue'], ['red', 'red'], ['green', 'green'], ['magenta', 'magenta'], ['cyan', 'cyan'],
             ['orange', 'orange'], ['olive', 'olive']]

cols = ['current_step', 'population', 'hh_income', 'gini', 'gdp', 'wealth_gini', 'unemployment',
        'housing_price_indexes', 'n_matching_housing', 'tom_buy', 'tom_sell',
        'total_posted_jobs', 'skill', 'n_matching_labour', 'recruit_outside', 'tom_labour']


excluded_list = ['current_step', 'n_posted_jobs_unemployed', 'n_applicants_unemployed']

for i in list(params.INDUSTRIES['name']):
    cols += [f'person_income_{i}']
    cols += [f'skill_{i}']
    cols += [f'n_posted_jobs_{i}']
    cols += [f'n_applicants_{i}']
    cols += [f'n_workers_{i}']


def averaging_plotting(path, to_eliminate=0, multiple_runs=False):
    print('Plotting averages...')
    if not multiple_runs:
        folders = os.listdir(path)
    else:
        folders = [path]
    for each in folders:
        if each is path:
            path2 = path
        else:
            path2 = os.path.join(path, each)
        geos = [g for g in os.listdir(path2) if g.endswith('.shp')]
        tables = [f for f in os.listdir(path2) if f.endswith('.csv')]
        d = pd.DataFrame(columns=cols)
        g = gpd.GeoDataFrame(columns=['Name', 'income', 'avghprcs', 'p_skilled', 'njobsnbhd'])
        for table in tables:
            path3 = os.path.join(path2, table)
            d = d.append(pd.read_csv(path3))
        for geo in geos:
            path3 = os.path.join(path2, geo)
            g = g.append(gpd.read_file(path3))
        for col in d.columns:
            d[col] = d[col].astype(float)
        avg_table = d.groupby(by='current_step').agg('mean').reset_index()

        plt_path = os.path.join(path2, 'avg')
        if not os.path.exists(plt_path):
            os.mkdir(plt_path)
        plotting.plotting(avg_table, plt_path, to_eliminate)
        print('Sensivity done!')


def sensitivity(prs, param=None, values=None, n_times=1, scenario=False, n_cpus=1, scene_params=None, verbose=False):
    # Creating folder for each sensitivity analysis to centralize average plots
    if not param:
        sensitivity_path = f'multiple_{n_times}_run'
        multiple_runs = True
    else:
        sensitivity_path = param
        multiple_runs = False
        if not scene_params:
            scene_params = prs.copy()
    if not os.path.exists(prs['SAVING_DIRECTORY']):
        os.mkdir(prs['SAVING_DIRECTORY'])
    new_path = os.path.join(prs['SAVING_DIRECTORY'], sensitivity_path)
    if os.path.exists(new_path):
        print('ADDING NEW RUNS ON TOP OF EXISTING RESULTS')
    else:
        os.mkdir(new_path)
    prs['SAVING_DIRECTORY'] = new_path
    scene_params['SAVING_DIRECTORY'] = new_path

    # Actual sensitivity analysis in Parallel
    if param and not scenario:
        with Parallel(n_jobs=n_cpus) as parallel:
            parallel(delayed(simulation.main)(prs, param, v, seed=k) for v in values for k in range(n_times))
        # Averaging and Plotting
        averaging_plotting(new_path, to_eliminate=prs['TIME_TO_BE_ELIMINATED'], multiple_runs=multiple_runs)
    elif not scenario:
        with Parallel(n_jobs=n_cpus) as parallel:
            parallel(delayed(simulation.main)(prs, seed=k, verbose=verbose) for k in range(n_times))
        # Averaging and Plotting
        averaging_plotting(new_path, to_eliminate=prs['TIME_TO_BE_ELIMINATED'], multiple_runs=True)
    elif param and scenario:
        print(f'n_times: {n_times} n_cpus: {n_cpus}')
        with Parallel(n_jobs=n_cpus) as parallel:
            parallel(delayed(simulation.main)(prs, seed=k, scene_params=scene_params) for k in range(n_times))


def sensitivity_plotting(path):
    print('Plotting sensitivity comparison...')
    database = defaultdict(dict)
    scene_dirs = to_dict_from_module(scenes)
    folders_list = [f for f in os.listdir(path)
                    if f not in [key for key in scene_dirs['SCENARIOS']]
                    and os.path.isdir(os.path.join(path, f))]

    # Getting the data
    # All parameters tested
    for each in folders_list:
        if os.path.exists(os.path.join(path, each, 'plots')):
            print(f"Plots already exist for this parameter: {each}.")
            print(f"If you want new plots, please delete 'plots' folder")
            continue
        path2 = os.path.join(path, each)
        runs = [f for f in os.listdir(path2) if '=' in f]
        if runs:
            database[each] = defaultdict(dict)
            # Variations within parameters
            for par in runs:
                d = pd.DataFrame(columns=cols)
                value = par.split('=')[1]
                path3 = os.path.join(path2, par)
                tables = [f for f in os.listdir(path3) if f.endswith('.csv')]
                if tables:
                    for table in tables:
                        t = pd.read_csv(os.path.join(path3, table))
                        if not t.empty:
                            d = d.append(t)
                    for col in d.columns:
                        d[col] = pd.to_numeric(d[col])
                    database[each][value]['upper_table'] = d.groupby(by='current_step').quantile(.95).reset_index()
                    database[each][value]['lower_table'] = d.groupby(by='current_step').quantile(.05).reset_index()
                    database[each][value]['avg_table'] = d.groupby(by='current_step').agg('mean').reset_index()

            # # Plotting comparisons
            for col in cols:
                if col not in excluded_list:
                    for key in database:
                        fig, ax = plt.subplots()
                        for j, value in enumerate(database[key]):

                            ax.plot(database[key][value]['avg_table']['current_step'],
                                    database[key][value]['avg_table'][col],
                                    color=color_map[j][1])
                            ax.plot(database[key][value]['lower_table']['current_step'],
                                    database[key][value]['lower_table'][col],
                                    color=color_map[j][0], alpha=.5)
                            ax.plot(database[key][value]['upper_table']['current_step'],
                                    database[key][value]['upper_table'][col],
                                    color=color_map[j][0], alpha=.5)
                            ax.fill_between(database[key][value]['avg_table']['current_step'],
                                            database[key][value]['upper_table'][col],
                                            database[key][value]['lower_table'][col],
                                            facecolor=color_map[j][1], alpha=.3, label=value)
                        path3 = os.path.join(path2, 'plots')
                        if not os.path.exists(path3):
                            os.mkdir(path3)
                        plotting.finish_plot(ax, col, path3)
                        plt.close()
    print('Sensitivity plotting done!')


def scenario_plotting(prs, scenarios, single_scenario=None):
    print('Plotting scenarios...')
    path = os.path.join(prs['SAVING_DIRECTORY'], 'scenarios_plots')
    if not os.path.exists(path):
        os.mkdir(path)
    database = defaultdict(dict)
    if not single_scenario:
        folders = list(scenarios['SCENARIOS'].keys())
    else:
        folders = [key for key in single_scenario]

    # Getting the data
    for each in folders:
        path2 = os.path.join(prs['SAVING_DIRECTORY'], each)
        tables = [f for f in os.listdir(path2) if f.endswith('.csv')]
        d = pd.DataFrame(columns=cols)
        for table in tables:
            path3 = os.path.join(path2, table)
            d = d.append(pd.read_csv(path3))
        for col in d.columns:
            d[col] = pd.to_numeric(d[col])
        database[each]['upper_table'] = d.groupby(by='current_step').quantile(.86).reset_index()
        database[each]['lower_table'] = d.groupby(by='current_step').quantile(.16).reset_index()
        database[each]['avg_table'] = d.groupby(by='current_step').agg('mean').reset_index()

    # Plotting comparisons
    for col in cols:
        if col != 'current_step':
            fig, ax = plt.subplots()
            for i, each in enumerate(folders):
                ax.plot(database[each]['avg_table']['current_step'], database[each]['avg_table'][col],
                        color=color_map[i][0],
                        label=each)
                ax.plot(database[each]['lower_table']['current_step'], database[each]['lower_table'][col],
                        color=color_map[i][0],
                        alpha=.5)
                ax.plot(database[each]['upper_table']['current_step'], database[each]['upper_table'][col],
                        color=color_map[i][0],
                        alpha=.5)
                # ax.fill_between(database[each]['avg_table']['current_step'],
                #                 database[each]['upper_table'][col],
                #                 database[each]['lower_table'][col],
                #                 facecolor=color_map[i][0], alpha=0.1, label=each)

            plotting.finish_plot(ax, col, path)
            plt.close()
    print('Scenarios plotted done!')


def main(prs):
    return simulation.main(prs)


def to_dict_from_module(module):
    return {k: getattr(module, k) for k in dir(module) if not k.startswith('_')}


def update(d, u):
    """ Updates dictionaries of varying lengths.
        Enter original dict, then updating data, returns new updated dictionary
        """
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def run_scenarios(prs, n_cpus, n_runs, scenarios, single_scenario=None):
    if not single_scenario:
        single_scenario = list(scenarios['SCENARIOS'].keys())
    original = copy.deepcopy(prs)
    for key in scenarios['SCENARIOS']:
        if key in single_scenario:
            scene_params = update(prs, scenarios['SCENARIOS'][key]).copy()
            prs = copy.deepcopy(original)
            print(f'Scene params: {key}')
            sensitivity(prs, param=key, n_times=n_runs, n_cpus=n_cpus, scenario=True, scene_params=scene_params)
    # 3. Plot both results.
    scenario_plotting(original, scenarios, single_scenario)


if __name__ == '__main__':
    t0 = time.time()
    # Read parameters and transform into a dictionary
    prsd = to_dict_from_module(params)
    # Scenarios
    s = to_dict_from_module(scenes)
    # Number of runs
    m = 1
    # Number of cpus (cores of the machine)
    cpus = 1
    verbis = False

    # STANDARD procedure:
    # <python   main.py
    #           NUMBER_CPUS/plotting
    #           NUMBER_RUNS/names of plotting folders (includes sensitivity), default folder: output, for example
    #           SENSITIVITY/SCENARIOS/RUN
    #           ALL/SCENARIOS SEPARATED BY SPACE/PARAM VALUES>
    if len(sys.argv) == 1:
        print('Running simple run...')
        sensitivity(prsd, n_times=m, n_cpus=cpus, scene_params=s, verbose=True)
    elif sys.argv[1].lower() == 'plotting':
        if len(sys.argv) == 2:
            print(f"Just plotting scenarios for path: {prsd['SAVING_DIRECTORY']}")
            scenario_plotting(prsd, s)
        elif len(sys.argv) == 3:
            print(f'Plotting sensitivity from path {sys.argv[2]}')
            sensitivity_plotting(sys.argv[2])
        else:
            ss = [s for s in sys.argv[2:]]
            print(f"Just plotting scenarios for paths: {ss}")
            scenario_plotting(prsd, s, ss)
    else:
        try:
            cpus = int(sys.argv[1])
            m = int(sys.argv[2])
        except IndexError:
            print('Please, enter integers to refer to number of CPUS and number of runs and choice of run as in:'
                  '<python main.py 1 1 run> or <python main.py 1 1 scenarios ALL> or \n'
                  '<python main.py 1 1 sensitivity JOB_SEARCH_SIZE [10,11,12,14]>')
        if sys.argv[3].lower() == 'run' or len(sys.argv) == 3:
            print(f'Running {m} runs...')
            if m == 1:
                verbis = True
            sensitivity(prsd, n_times=m, n_cpus=cpus, scene_params=s, verbose=verbis)
        elif sys.argv[3].lower() == 'sensitivity':
            try:
                vs = list(map(float, sys.argv[5].strip('[]').split(',')))
                print(f'Running sensitivity analysis with parameters {vs}...')
                sensitivity(prsd, param=sys.argv[4], values=vs, n_times=m, n_cpus=cpus)
                try:
                    sensitivity_plotting(prsd['SAVING_DIRECTORY'])
                    print(f"Plotting sensitivity analysis for each paramater at {prsd['SAVING_DIRECTORY']}")
                except:
                    print('Could not get correct path for plotting sensitivity')
            except IndexError:
                print('Please enter parameter and values to test. Values should be in the format:'
                      ' [2,3,4,5] without spaces\n'
                      'Example: python main.py 1 1 sensitivity JOB_SEARCH_SIZE [10,11,12,14]')
                sys.exit(1)
        elif sys.argv[3].lower() == 'scenarios':
            try:
                if sys.argv[4].upper() == 'ALL':
                    print(f"Running ALL scenarios: ({list(s['SCENARIOS'].keys())})")
                    run_scenarios(prsd, n_cpus=cpus, n_runs=m, scenarios=s)
                    scenario_plotting(prsd, s)
                else:
                    ss = [sc for sc in sys.argv[4:]]
                    print(f'Running scenarios {ss}')
                    run_scenarios(prsd, n_cpus=cpus, n_runs=m, scenarios=s, single_scenario=ss)
                    scenario_plotting(prsd, s)
            except IndexError:
                print('No scenarios described, running for ALL scenarios')
                run_scenarios(prsd, n_cpus=cpus, n_runs=m, scenarios=s)
                scenario_plotting(prsd, s)

    print(f'This run took {time.time() - t0:.2f} seconds')
