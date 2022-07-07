# There is  a list of jobs available, and people apply for them, the highest qualified person gets the job
import pandas as pd


class LabourMarket:
    def __init__(self, model=None):
        self.jobs = list()
        self.jobs_data = pd.DataFrame(columns=['Name', 'n_jobs', 'step'])
        self.model = model

    def post_job(self, job):
        self.jobs.append(job)

    def match_process(self):
        self.step_reporting()
        n_recruits = 0
        # Match between jobs and applicants
        for job in self.jobs:
            if job.applicants:
                # Sort candidates by skill
                job.applicants.sort(reverse=True, key=lambda w: w.skill)
                for i in range(len(job.applicants)):
                    candidate = job.applicants[i]
                    # candidate choose the first job offer made to him
                    if candidate.employer is None and candidate.skill > job.min_skill:
                        job.employer.hire(candidate, job.income)
                        candidate.household.renter_move()
                        n_recruits += 1
                        break
        self.jobs = list()  # all jobs deleted in the previous step deleted
        self.model.reporter.loc[self.model.current_step, 'n_matching_labour'] = n_recruits

    def collect_jobs_data(self):
        idx = len(self.jobs_data)
        # Keeping neighbourhood data of jobs posted
        nbhds = [n.employer.neighbourhood for n in self.jobs]
        nbhds = {n: nbhds.count(n) for n in nbhds}
        for nbhd in nbhds:
            self.jobs_data.loc[idx, 'n_jobs'] = nbhds[nbhd]
            self.jobs_data.loc[idx, 'step'] = self.model.current_step
            self.jobs_data.loc[idx, 'Name'] = nbhd.name
            idx += 1

    def process_data(self, last_step):
        results = self.jobs_data[self.jobs_data['step'] > last_step].copy()
        results.loc[:, 'n_jobs'] = results['n_jobs'].astype(int)
        return results.groupby(by='Name').agg('mean')

    def step_reporting(self):
        self.collect_jobs_data()
        self.model.reporter.loc[self.model.current_step, 'total_posted_jobs'] = len(self.jobs)
        industry_names = list((self.model.params['INDUSTRIES']['name']))
        for i in industry_names:
            jobs_in_industry = [j for j in self.jobs if j.industry.name == i]
            self.model.reporter.loc[self.model.current_step, f'n_posted_jobs_{i}'] = len(jobs_in_industry)
            n_applicants = self.model.np.sum([len(job.applicants) for job in jobs_in_industry])
            self.model.reporter.loc[self.model.current_step, f'n_applicants_{i}'] = n_applicants


class Job:
    def __init__(self, employer, income=0.0, min_skill=0.0):  # employer class: Businesses
        self.employer = employer
        self.income = income
        self.location = employer.location  # business location
        self.min_skill = min_skill
        self.industry = employer.industry
        self.applicants = list()  # list of applicants
