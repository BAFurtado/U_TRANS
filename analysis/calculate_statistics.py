

def calculate_unemployment(model):
    unemployed = len([p for p in model.persons if p.industry.name == "unemployed"])
#    active = len([p for p in model.persons if not p.retired])
    return round(unemployed / len(model.persons), 4)


def calculate_gini(incomes, model):
    # Sort smallest to largest
    cumm = model.np.sort(incomes)
    # Values cannot be 0
    cumm += .00001
    # Find cumulative totals
    n = cumm.shape[0]
    index = model.np.arange(1, n + 1)
    gini = ((model.np.sum((2 * index - n - 1) * cumm)) / (n * model.np.sum(cumm)))
    return gini


def calculate_gdp(incomes):
    return sum(incomes)

