"""

Service industries include public:
 administration,
 human health,
 education,
 facilities (such as retail and
catering), social services and so on.

Non-service industries do not serve local households directly, including:
 oil and gas, business to business services, agriculture, and manufacturing.

"""


class Industry:
    def __init__(self, name='', size=0.0, skill_min=0.0, income_min=0.0,
                 income_max=0.0, growth_rate_mean=0, growth_rate_sd=0, model=None):
        self.name = name
        self.size = size  # percentage of employment in the population
        self.skill_min = skill_min
        self.income_min = income_min  # income: lognormal distributions
        self.income_max = income_max
        self.growth_rate_mean = growth_rate_mean
        self.growth_rate_sd = growth_rate_sd
        self.model = model

    def update_growth_rate(self):
        # Growth rate is the second element (index 1) of a tuple. First is the step of going into effectiveness
        try:
            self.growth_rate_mean = \
                self.model.params['INDUSTRIES']['growth_rate_mean'][self.name]['step-rate'][self.model.current_step] \
                if self.name in self.model.params['INDUSTRIES']['growth_rate_mean'] else 0
        except KeyError:
            pass
        # Updating only the mean, not the sd
        # self.growth_rate_sd = \
        #     self.model.params['INDUSTRIES']['growth_rate_sd'][self.name]['step-rate'][self.model.current_step] \
        #     if self.name in self.model.params['INDUSTRIES']['growth_rate_sd'] else 0

    def draw_income(self):
        return self.model.seed.uniform(self.income_min, self.income_max)

    def __repr__(self):
        return str(self.name)
