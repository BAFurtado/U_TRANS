import os
import sys
import time

parameters_to_test = {
    # 'N_PEOPLE': '[5000,10000,15000,20000]',
    # 'N_HOUSES': '[1400,3000,4500,6000]',
    # 'PERC_ENTERING_MODEL': '[.0001,.0002,.00031,.0005]',
    # 'PERC_LEAVING_RANDOM': '[.02,.01,.005,.001]',  # per year
    # 'PERC_LEAVING_JOB_OUTSIDE': '[.3,.2,.1,.05]',  # per year, a measure of brain drain
    # 'PERC_LEAVING_UNEMPLOYED': '[.03,.02,.01,.05]'  # per year, so unemployment does not go through the roof
    # 'PERC_TRAINING': '[.005,.01,.015,.025]',
    # 'N_JOBS_APPLIED': '[1,2,3,10]',  # how flexible the matching process in the labor market
    # 'JOB_SEARCH_SIZE': '[1,3,5,10]',
    # 'SKILLED_THRESHOLD': '[.3,.5,.7]',
    # 'SKILLED_DECAY_RATE': '[.0005,.001,.002,.003]',
    # 'SKILL_GROWTH_TRAINING': '[.001,.003,.005,.01]',  # PER WEEK
    # 'N_WEEKS_BEFORE_SKILL_DECAY': '[13,26,52]',
    # 'MAX_BUSINESS_TOM': '[1,2,4,10]',  # after that recruit outside
    # 'INITIAL_SKILL_VARIANCE': '[.01,.02,.04]',
    # 'MAX_DISTANCE_TO_JOB': '[1000,5000,10000,20000]',
    # 'DELTA_INCOME_SELL_HOUSE': '[-.1,-.3,-.5]',
    # 'ASK_PREMIUM': '[.05,.1,.2]',
    # 'ASK_DECREASE_PER_WEEK': '[.005,.01,.02]',
    # 'BID_DISCOUNT': '[.001,.005,.1,.15]',
    # 'BID_INCREASE_PER_WEEK': '[.005,.01,.15]',
    # 'MAX_HOUSE_PRICE_TO_ANNUAL_HH_INCOME': '[4,7,10,20]',
    # 'REFERENCE_PRICE_PERIOD': '[4,6,8,10]',
    # 'P_RENTER_BUY': '[0.1,0.3,0.5]',
    # 'P_RANDOM_SELL': '[.005,.01,.15,.2]',  # PER YEAR
    # 'N_HOUSES_TO_EVALUATE': '[5,10,20,40]',
    # 'MAX_TOM_HOUSING': '[1,10,26,52]',  # half a year
    # 'VAR_INIT_PRICE': '[10,25,50,60]'
    # 'NBHD_RANDOM_FACTOR_RANGE': '[0.1,0.3,0.5]',
    # 'MEAN_WEIGHT_DISTANCE': '[.45,0.6,.75]',
    # 'MEAN_WEIGHT_INCOME': '[.45,0.6,.75]',
    # 'MEAN_WEIGHT_PRICE': '[.1,0.2,.3]',
    # 'MEAN_WEIGHT_SERVICE': '[.1,0.2,.3]',
    # 'MEAN_WEIGHT_SKILL': '[.1,0.2,.3]',
    # 'MEAN_WEIGHT_UNEMPLOYMENT': '[.45,0.6,.75]',
    # 'MIN_HOUSE_SCORE': '[.65,.73,.78,.83,.88]',
    'MIN_HOUSE_SCORE_SD': '[.11,.15,.18,.22,.25]'
    }


def main(cpus=10, runs=4):
    with open('errors_on_sensitivity.txt', 'a') as handler:
        for p in parameters_to_test:
            comm = f'python main.py {cpus} {runs} sensitivity {p} {parameters_to_test[p]}'
            print(comm)
            try:
                os.system(comm)
            except:
                handler.write(f'{time.asctime()}: \n{comm} \n')


if __name__ == '__main__':
    c = 12
    r = 40
    if len(sys.argv) == 3:
        c = int(sys.argv[1])
        r = int(sys.argv[2])
        main(c, r)
    else:
        main(c, r)
