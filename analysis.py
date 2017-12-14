# Hypothesis Testing Demo
# Code by Dan Truong

# Experimental Context: look at a hypothetical scenario where the effect of 
# ascorbic acid is evaluated against the common cold. The data is based on a 
# 1961 experiment where 279 French skiers were given either ascorbic acid or 
# a placebo. After two weeks, the skiers were examined to determine if they 
# still had symptoms of the cold. 

# Dependent Libraries
import pprint
import math
from scipy import stats

# Observed Distribution Array
obs = {
    "a" : 17, ## Ascorbic Acid Treatment, Cold Symptoms 
    "b" : 31, ## Placebo Treatment, Cold Symptoms
    "c" : 122, ## Ascorbic Acid Treatment, No Cold Symptoms
    "d" : 109 ## Placebo Treatment, No Cold Symptoms
}

n = sum(obs.values()) ## Sum of participants

aEXP = (obs['a'] + obs['b']) * (obs['a'] + obs['c']) / n
bEXP = (obs['a'] + obs['b']) * (obs['b'] + obs['d']) / n
cEXP = (obs['c'] + obs['d']) * (obs['a'] + obs['c']) / n
dEXP = (obs['c'] + obs['d']) * (obs['b'] + obs['d']) / n

# Expected Distribution Array
exp = {
    "Ascorbic Acid Treatment, Cold Symptoms" : aEXP,
    "Placebo Treatment, Cold Symptoms" : bEXP,
    "Ascorbic Acid Treatment, No Cold Symptoms" : cEXP,
    "Placebo Treatment, No Cold Symptoms" : dEXP
}
pprint.PrettyPrinter(indent = 3).pprint(exp)

# Chi-squared Value

arrVals = [
    abs(obs['a'] - aEXP) ** 2 / aEXP,
    abs(obs['b'] - bEXP) ** 2 / bEXP,
    abs(obs['c'] - cEXP) ** 2 / cEXP,
    abs(obs['d'] - dEXP) ** 2 / dEXP,
]
print("Chi-squared value: " + str(sum(arrVals)))

# P-value (with 1 DF)
pValueChiSq = 1 - stats.chi2.cdf(sum(arrVals) , 1) 
print("P-value: " + str(pValueChiSq))

# Fisher's Exact Test
oddsRatio, pValueFET = stats.fisher_exact([
    [obs['a'], obs['c']], 
    [obs['b'], obs['d']]
], alternative = "less")
print("P-value: " + str(pValueFET))

# Difference between Chi-squared and FET p-values
print(abs(pValueChiSq - pValueFET))


# Relative Risk and Odds Ratio

# For the purpose of clarity, we’ll define the following variables:
# 	Exposed: treatment by Ascorbic Acid
# 	Nonexposed: treatment by Placebo
# 	Disease: presence of cold symptoms after 2 weeks
# 	Nondisease: non-presence of cold symptoms

# We will also redefine the variables 'a', 'b', 'c', & 'd' as follows:
# 	a = Exposed group with diseased outcome (17)
# 	b = Exposed group with non-diseased outcome (122)
# 	c = Nonexposed group with diseased outcome (31)
# 	d = Nonexposed group with non-diseased outcome (109)

a = obs['a']
b = obs['c']
c = obs['b']
d = obs['d']

# Relative Risk Ratio
rr = (a / (a + b)) / (c / (c + d))

# Standard Error of ln(Relative Risk)
se = math.sqrt(1 / a + 1 / c - 1 / (a + b) - 1 / (c + d))

# 95% Confidence Interval
lowBound = math.exp(math.log(rr) - 1.96 * se)
highBound = math.exp(math.log(rr) + 1.96 * se)

# Standard Normal Deviate
zScore = math.log(rr) / se

# Two-tailed P-value
pValue = stats.norm.sf(abs(zScore)) * 2

print("Relative Risk Ratio: " + str(rr))
print("95% Confidence Interval: [" 
      + str(lowBound) + ", " + str(highBound) + "]")
print("P-value: " + str(pValue))

# Odds Ratio
odds = (a * d) / (b * c)

# Standard Error of ln(Odds Ratio)
se = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)

# 95% Confidence Interval
lowBound = math.exp(math.log(odds) - 1.96 * se)
highBound = math.exp(math.log(odds) + 1.96 * se)

# Standard Normal Deviate
zScore = math.log(odds) / se

# Two-tailed P-value
pValue = stats.norm.sf(abs(zScore)) * 2

print("Odds Ratio: " + str(odds))
print("95% Confidence Interval: [" 
      + str(lowBound) + ", " + str(highBound) + "]")
print("P-value: " + str(pValue))