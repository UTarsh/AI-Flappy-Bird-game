import pygame
import time
import os
import random
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle 
import seaborn as sb



'''
Graph 1
Average Score and Highest Score - for best parameters
'''
df = pd.read_csv('data_collected/avg-highest.csv')
df.set_index("Score", inplace = True) 
print(df)
# df = df.fillna(100)

gen_list = []
for i in range(0,23):
    gen_list.append("Gen "+str(i))
df = df[gen_list]
df=df.T

print(df)
clrs = ['tab:blue', 'tab:red' ]

df.plot(linewidth=1,marker='h',markersize = 4, color = clrs)

plt.xlabel("Generation")
plt.ylabel("Score")
plt.show()

'''
Graph 2 -  Average Score progression for different mutation rates
'''
df = pd.read_csv('data_collected/mutation-rate-variation-2.csv')
df = df[-4:]
df.set_index("Mutation Rate", inplace = True) 
print(df)
df = df.fillna(100)

gen_list = []
for i in range(0,40):
    gen_list.append("Gen "+str(i))
df = df[gen_list]
df=df.T

df.plot(linewidth=1)
plt.xlabel("Generation")
plt.ylabel("Average Score Per Generation - 12")
plt.show()

'''
Graph 3- bar Graph - Mutation Rates
'''

mut_high = {'0.001': 2, '0.005': 2, '0.01': 2, '0.05': 8, '0.1': 0, '0.2': 2, '0.3': 6, '0.5': 4, '0.9': 7}

a_file = open("data.pkl", "wb")
pickle.dump(mut_high, a_file)
a_file.close()

keys = mut_high.keys()
values = mut_high.values()
for key in keys:
    mut_high[key]+=1
clrs = ['tab:blue' if (x >min(values)) else 'tab:green' for x in values ]

plt.bar(keys, values,color=clrs)
plt.xlabel("Mutation Rate")
plt.xticks(rotation=0)
plt.ylabel("Number of Generations required to reach 100 Score")

plt.show()

'''
Graph 4 - Game Difficulty Variation
'''
df = pd.read_csv('data_collected/gamedifficulty.csv',index_col=0)
gen_list = []
for i in range(0,50):
    gen_list.append("Gen "+str(i))
df = df[gen_list]
df=df.T

print(df)
df.plot()
plt.xlabel("Generation")
plt.ylabel("Average Score Per Generation")
plt.show()

'''
Graph5 - Variation of Average score for different Init pop size 
'''
df = pd.read_csv('data_collected/popsize.csv',index_col=0)
gen_list = []
for i in range(0,50):
    gen_list.append("Gen "+str(i))
df = df[gen_list]
df=df.T
print(df)
df.plot()
plt.xlabel("Generation")
plt.ylabel("Average Score Per Generation")
plt.show()

'''
Graph 6 - Variation of Average Score for Crossover techniques
'''
df = pd.read_csv('data_collected/crossover.csv',index_col=0)

gen_list = []
for i in range(0,50):
    gen_list.append("Gen "+str(i))
df = df[gen_list]
df=df.T

print(df)
df.plot()
plt.xlabel("Generation")
plt.ylabel("Average Score Per Generation")
plt.show()




