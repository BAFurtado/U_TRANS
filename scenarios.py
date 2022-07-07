GRACE_PERIOD = 100

SCENARIOS = {'BASELINE':  # baseline should be equal to params [INDUSTRIES]
                 {'INDUSTRIES':
                      {'growth_rate_mean':
                           {'old': {'step-rate': {0: 0}},  # per period/week
                            'new': {'step-rate': {0: 0}},  # per period/week
                            },
                       },
                  'SCENARIO': 'BASELINE'
                  },
             'TRANSITION':
                 {'INDUSTRIES':
                      {'growth_rate_mean':
                           {'old': {'step-rate': {0: 0, GRACE_PERIOD: -.002}},  # per period/week
                            'new': {'step-rate': {0: 0, GRACE_PERIOD: .002}},  # per period/week
                            },
                       },
                  'SCENARIO': 'TRANSITION'
                  },
             'TRANSITION_TRAINING':
                 {'INDUSTRIES':
                      {'growth_rate_mean':
                           {'old': {'step-rate': {0: 0, GRACE_PERIOD: -.002}},  # per period/week
                            'new': {'step-rate': {0: 0, GRACE_PERIOD: .002}},  # per period/week
                            },
                       },
                  'P_TRAINING': .1,
                  'SCENARIO': 'TRANSITION_TRAINING'
                  },
             'TOTAL_COLLAPSE':
                 {'INDUSTRIES':
                      {'growth_rate_mean':
                           {'old': {'step-rate': {0: 0, GRACE_PERIOD: -.01}},  # per period/week,  # per period/week
                            'new': {'step-rate': {0: 0, GRACE_PERIOD: .0005}},  # per period/week
                            },
                       },
                  'SCENARIO': 'TOTAL_COLLAPSE'
                  },
             'WEAK_RECOVERY':
                 {'INDUSTRIES':
                      {'growth_rate_mean':
                           {'old': {'step-rate': {0: 0, GRACE_PERIOD: -.002}},  # per period/week
                            'new': {'step-rate': {0: 0, GRACE_PERIOD: .0005}},  # per period/week
                            },
                       },
                  'SCENARIO': 'WEAK_RECOVERY'
                  },
             'OVERSHOOTING':
                 {'INDUSTRIES':
                      {'growth_rate_mean':
                           {'old': {'step-rate': {0: 0, GRACE_PERIOD: -.01}},  # per period/week
                            'new': {'step-rate': {0: 0, GRACE_PERIOD: .005}}
                            },
                       'income_max': {'new': 1000}
                       },
                  'SCENARIO': 'OVERSHOOTING'
                  },
             'AGGRESSIVE_EXPANSION':
                 {'INDUSTRIES':
                      {'growth_rate_mean':
                           {'old': {'step-rate': {0: 0, GRACE_PERIOD: -.01}},  # per period/week  100 is GRACE_PERIOD
                            'new': {'step-rate': {0: 0, GRACE_PERIOD: .005, GRACE_PERIOD + 200: .001}}
                            },
                       'income_max': {'new': 1000}
                       },
                  'SCENARIO': 'AGGRESSIVE_EXPANSION'
                  },
             }
