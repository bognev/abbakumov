from scipy.stats import norm, chi2_contingency
import statsmodels.api as sm
import numpy as np
#AB testing
s1 = 135
n1 = 1781
s2 = 47
n2 = 1443
p1 = s1/n1
p2 = s2/n2
p = (s1+s2)/(n1+n2)
z = (p2-p1)/((p*(1-p)*((1/n1)+(1/n2)))**0.5) #z-метка
p_value = norm.cdf(z)
print(['{:.12f}'.format(a) for a in (abs(z), p_value*2)])
z1, p_value1 = sm.stats.proportions_ztest([s1, s2], [n1, n2])
print(['{:.12f}'.format(a) for a in (abs(z1), p_value1*2)])

arr = np.array([[s1, n1-s1],[s2, n2-s2]])
chi2, p_value2, dof, exp = chi2_contingency(arr, correction=False)
print(['{:.12f}'.format(a) for a in (chi2**0.5, p_value2)])