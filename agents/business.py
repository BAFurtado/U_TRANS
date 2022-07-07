from markets.labourMarket import Job


class Business:

    def __init__(self, _id, industry=None, target_size=0.0, location=None, neighbourhood=None, model=None):
        self.id = _id
        self.industry = industry
        self.location = location  # shapely point
        self.neighbourhood = neighbourhood
        self.target_size = target_size  # double
        self.workers = list()  # person dictionary
        self.model = model
        self.growth_rate = 0.0
        self.tom = 0.0 # n periods looking for a recruit

    def post_job(self, min_skill, income):
        job = Job(self, income=income, min_skill=min_skill)
        self.model.labour_market.post_job(job)

    def make_recruit_decision(self):
        n_vacancy = round(self.target_size - len(self.workers))
        if n_vacancy > 0:
            for i in range(n_vacancy):
                job_min_skill = self.industry.skill_min
                income = self.industry.draw_income()
                self.post_job(job_min_skill, income)
        elif n_vacancy < 0:
            n_redundancy = -1 * n_vacancy
            # Randomly fire n workers
            # print(f'before fire, n workers = {len(self.workers)}, n redundancy = {n_redundancy}')
            [self.fire(worker=w)
             for w in self.model.seed.choice(self.workers, size=n_redundancy, replace=False)]
            # print(f'after fire, n workers = {len(self.workers)}')

    def update_target_size(self):
        self.update_growth_rate()
        self.target_size = self.target_size * (1 + self.growth_rate)

    # after the match process in the labour mkt, if businesses cannot meet target size, recruit outside
    def recruit_outside(self, n):
        new_recruits = self.model.add_new_persons(n=n, industry=self.industry)
        self.model.add_new_households(new_recruits)
        for person in new_recruits:
            self.hire(person, self.industry.draw_income())

    def update_growth_rate(self):
        if self.industry.name == "service":
            self.growth_rate = 0
            # if len(self.neighbourhood.data['n_households']) >= 2:
            #     if self.neighbourhood.data['n_households'][-2] > 1:
            #         self.growth_rate = 0
            #         # self.growth_rate = self.neighbourhood.data['avg_income'][-1] *
            #         # self.neighbourhood.data['n_households'][-1] /
            #         # (self.neighbourhood.data['avg_income'][-2] * self.neighbourhood.data['n_households'][-2]) - 1
            #     else:
            #         self.growth_rate = 0
            # else:
            #     self.growth_rate = 0
        else:
            self.growth_rate = self.model.seed.normal(self.industry.growth_rate_mean, self.industry.growth_rate_sd)

    def hire(self, worker, income):
        self.workers.append(worker)
        worker.employer = self
        worker.industry = self.industry
        worker.TOM = 0
        worker.income = income

    def fire(self, worker):
        self.workers.remove(worker)
        worker.employer = None
        worker.industry = self.model.industries['unemployed']
        worker.income = worker.industry.draw_income()

    def move(self, location):
        self.location = location
