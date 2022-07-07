""" The person of the model

1. Methods. be born in python would be the __init__ function
2. Die method will be implemented within the Household to which the person is a member.

employed: part-time employee, full-time employee, self-employed,
inactive: unemployed, retired, student, looking after home of family, and long-term sick or disabled

info comes as :males/females aged 16 to 34, 35–54, 55–74, and 75 and over
"""


class Person:

    def __init__(self, _id=0, age=0, female=False, income=0.0, skill=0.0,
                 industry=None, employer=None, model=None):
        self.id = _id
        self.age = age
        self.female = female
        self.income = income
        self.skill = skill
        self.industry = industry
        # Here household will be the household object itself
        self.household = None
        self.experience = dict()
        self.partner = None
        self.employer = employer
        self.retired = False
        self.TOM = 0  # time on job market
        self.model = model
        self.random_week = self.model.seed.choice(range(1, (1 + model.params['WEEKS_PER_YEAR'])))

    def marry(self, other):
        self.partner = other
        other.partner = self

    def divorce(self):
        if self.partner:
            self.partner.partner = None
            self.partner = None

    def apply_for_job(self, job):
        job.applicants.append(self)

    def retire(self):
        self.retired = True

    # if unemployed, conduct job search
    def job_search(self):  # jm: job market
        size = min(len(self.model.labour_market.jobs), int(self.model.params['JOB_SEARCH_SIZE']))
        searched_jobs = self.model.seed.choice(self.model.labour_market.jobs, size=size, replace=False)
        eligible_jobs = [j for j in searched_jobs if self.skill >= j.min_skill]
        if eligible_jobs:
            # apply for n jobs with the highest income
            eligible_jobs.sort(key=lambda j: j.income, reverse=True)
            n = min(self.model.params['N_JOBS_APPLIED'], len(eligible_jobs))
            for job in eligible_jobs[0: int(n)]:
                self.apply_for_job(job)

    def update_tom(self):
        if self.industry.name == 'unemployed':
            self.TOM += 1

    def update_skill(self):
        if self.model.seed.uniform(0, 1) < self.model.params['P_TRAINING']:
            self.skill = min(1, self.skill * (1 + self.model.params['SKILL_GROWTH_TRAINING']))
        elif self.industry.name == 'unemployed' and self.TOM > self.model.params['N_WEEKS_BEFORE_SKILL_DECAY']:
            self.skill = max(0, self.skill * (1 - self.model.params['SKILL_DECAY_RATE']))

    def __str__(self):
        return f'Person {self.id} is a {self.age} year-old {"female" if self.female else "male"}.'
