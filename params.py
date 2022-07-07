# Abstract. Explain the phenomenon.
# Maybe have a table of ten other cities that contains minimum housing relocation prices in those cities plus a job
# demand in percentage.
# Creates an exogenous demand for a lock-in effect in the simulated city.

# RUN PARAMETERS
SAVING_DIRECTORY = 'output'
SHAPE_NAME = 'hexagons'
PLOT_EACH_RUN = False
PLOT_GE0_BUSINESS = False
SAVE_NBHD_SHPS = False
REPORT_NEIGHBOURHOOD_DATA = True
PLOT_NEIGHBOURHOOD_DATA = False
# The number of periods to be eliminated at the beginning
TIME_TO_BE_ELIMINATED = 0  # 0 * 52

SCENARIO = 'BASELINE'
GRACE_PERIOD = 100  # 0 * 52
WEEKS = 500  # 1 * 52

# CONSTANTS
MONTHS_PER_YEAR = 12
WEEKS_PER_YEAR = 52

# SIMULATION PARAMETERS
# DEMOGRAPHICS
# Working population between 20-60
MIN_AGE = 20
MAX_AGE = 60
PERC_MARRIED = .5
N_PEOPLE = 15000
N_HOUSES = 4500  # not including rentals
PERC_ENTERING_MODEL = 0.016 / 52  # per week
PERC_LEAVING_RANDOM = 0.01  # per year
PERC_LEAVING_JOB_OUTSIDE = 0.1  # per year, a measure of brain drain
PERC_LEAVING_UNEMPLOYED = 0.0  # per year, so unemployment does not go through the roof

# LABOUR MARKET
P_TRAINING = 0.01  # PERCENTAGE OF GETTING TRAINING PER PERIOD
N_JOBS_APPLIED = 3  # per period
JOB_SEARCH_SIZE = 10
SKILLED_THRESHOLD = 0.5  # [0,1], PERSON CONSIDERED SKILLED LABOUR IF SKILL ABOVE THRESHOLD
SKILL_DECAY_RATE = 0.001  # PER WEEK
SKILL_GROWTH_TRAINING = 0.005  # PER WEEK
N_WEEKS_BEFORE_SKILL_DECAY = 26
INITIAL_SKILL_VARIANCE = 0.2
MAX_DISTANCE_TO_JOB = 10000
MAX_BUSINESS_TOM = 4  # after that recruit outside

# HOUSING MARKET
DELTA_INCOME_SELL_HOUSE = -0.5
ASK_PREMIUM = 0.1
ASK_DECREASE_PER_WEEK = 0.01
BID_DISCOUNT = 0.1
BID_INCREASE_PER_WEEK = 0.01
MAX_HOUSE_PRICE_TO_ANNUAL_HH_INCOME = 10
REFERENCE_PRICE_PERIOD = 8
P_RENTER_BUY = 0.02  # PER year

P_RANDOM_SELL = 0.01  # PER YEAR
N_HOUSES_TO_EVALUATE = 20
MAX_TOM_HOUSING = 30  # weeks
VAR_INIT_PRICE = 50

MEAN_WEIGHT_DISTANCE = 0.6
MEAN_WEIGHT_INCOME = 0.6
MEAN_WEIGHT_PRICE = 0.2
MEAN_WEIGHT_SERVICE = 0.2
MEAN_WEIGHT_SKILL = 0.2
MEAN_WEIGHT_UNEMPLOYMENT = 0.6
# very sensitive to housing market and connected to mean_weight (so need to change together)
MIN_HOUSE_SCORE = 0.78
MIN_HOUSE_SCORE_SD = 0.18

# NBHD param
NBHD_RANDOM_FACTOR_RANGE = 0.1  # [0, 1] relative to nbhd_score, account for unobservable factors in choosing nbhds

INDUSTRIES = {'name': ['old', 'service', 'new', 'unemployed'],
              'n_businesses': {'old': 50,
                               'new': 50,
                               'service': 500,
                               'unemployed': 0},
              'initial_size': {'old': .4,  # size = perc of employment.
                               'new': .2,
                               'service': .3,
                               'unemployed': .1},
              'skill_min': {'old': .5,  # skill max = 1
                            'new': .5,
                            'service': 0,
                            'unemployed': 0},
              'income_max': {'old': 5,
                             'new': 5,
                             'service': 2,
                             'unemployed': 0},
              'income_min': {'old': 1,
                             'new': 1,
                             'service': 0,
                             'unemployed': 0
                             },
              'growth_rate_mean': {'old': {'step-rate': {0: 0}},
                                   'new': {'step-rate': {0: 0}}
                                   },  # per period/week
              'growth_rate_sd': {'old': {'step-rate': {0: 0}},
                                 'new': {'step-rate': {0: 0}}
                                 }
              }
