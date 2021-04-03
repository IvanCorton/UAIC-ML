from scipy.stats import binom

samples = 16
position = 12
prob = 0.5

#Probability to walk == 8 steps forward == 12 steps forward + 4 steps backward
binomi = binom.cdf(position, samples, prob)   # 0.9893646240234375

print (binomi)