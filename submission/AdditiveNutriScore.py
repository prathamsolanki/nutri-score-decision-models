print('### Executing UTA Approach ###')

import warnings
warnings.filterwarnings("ignore")

import csv
from pulp import *
import pandas as pd
import sys
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import sys

print('Importing the dataset...')

df = pd.read_excel("openfoodfacts_simplified_database.xlsx")
df = df[~df['nutrition_grade_fr'].isna()]

cols = ['energy_100g', 'saturated-fat_100g', 'sugars_100g', 'fiber_100g', 'proteins_100g', 'sodium_100g', 'nutrition_grade_fr']
df = df[cols]
df.columns = ['energy', 'saturated_fat', 'sugars', 'fiber', 'proteins', 'salt', 'nutriscore']

df = df[~df['energy'].isna()]
df = df[~df['saturated_fat'].isna()]
df = df[~df['sugars'].isna()]
df = df[~df['fiber'].isna()]
df = df[~df['proteins'].isna()]
df = df[~df['salt'].isna()]

def create_buckets(df, criterion, precision, eps):
    num_buckets = int((df[criterion].max() + eps - df[criterion].min()) / precision)
    max_value = df[criterion].max() + eps
    min_value = df[criterion].min()
    real_precision = (max_value - min_value) / num_buckets
    
    buckets = []
    left_thresh = min_value
    for i in range(num_buckets):
        buckets.append((left_thresh, left_thresh+real_precision))
        left_thresh = left_thresh+real_precision

    return buckets

buckets = {}

print('Creating the intervals...')

buckets['energy'] = create_buckets(df, 'energy', precision=200, eps=10)
buckets['saturated_fat'] = create_buckets(df, 'saturated_fat', precision=2, eps=0.1)
buckets['sugars'] = create_buckets(df, 'sugars', precision=4, eps=0.1)
buckets['fiber'] = create_buckets(df, 'fiber', precision=0.7, eps=0.1)
buckets['proteins'] = create_buckets(df, 'proteins', precision=2, eps=0.1)
buckets['salt'] = create_buckets(df, 'salt', precision=0.2, eps=0.05)

print('Splitting the dataset...')

features = ['energy', 'proteins', 'salt', 'fiber', 'saturated_fat', 'sugars']
X = df[features]
y = df[['nutriscore']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=1)

df = X_train
df['nutriscore'] = y_train['nutriscore']

df_test = X_test
df_test['nutriscore'] = y_test['nutriscore']

print('Formalizing the problem...')

prob = LpProblem("Nutriscore", LpMinimize)

sigma = {}

for index, food in df.iterrows():
  sigma[index] = LpVariable("sigma_"+str(index),0, sys.maxsize)

# The objective function is added to 'prob' first
prob += lpSum(list(sigma.values())), "Error in the ranking to be minimized"

utility_thresh = {}
for key in buckets:
    utility_thresh[key] = []
    
for key in buckets:    
    for i in range(len(buckets[key])):
        utility_thresh[key].append(LpVariable(key+"_"+str(i), 0, 1))
    utility_thresh[key].append(LpVariable(key+"_"+str(len(buckets[key])), 0, 1))

def get_bucket_index(value, buckets):
    return get_bucket_index_r(value, buckets, 0, len(buckets)-1)

def get_bucket_index_r(value, buckets, left, right):
    middle = int((right - left) / 2 + left)
    if (value >= buckets[middle][0]) and (value < buckets[middle][1]):
        return middle
    if value < buckets[middle][0]:
        return get_bucket_index_r(value, buckets, left, middle-1)
    return get_bucket_index_r(value, buckets, middle+1, right)

def utility_func(df, food_index):
    food = df.loc[food_index]
    criteria = list(food.keys())
    criteria.remove('nutriscore')
    utility = 0
    for criterion in criteria:
        bucket_index = get_bucket_index(food[criterion], buckets[criterion])
        left_thresh = buckets[criterion][bucket_index][0]
        right_thresh = buckets[criterion][bucket_index][1]
        m = (food[criterion] - left_thresh) / (right_thresh - food[criterion])
        left_utility = utility_thresh[criterion][bucket_index]
        right_utility = utility_thresh[criterion][bucket_index+1]
        
        utility += left_utility + m * (right_utility - left_utility)
    
    return utility

df['nutriscore'] = np.where(df['nutriscore']=='a', 5, df['nutriscore'])
df['nutriscore'] = np.where(df['nutriscore']=='b', 4, df['nutriscore'])
df['nutriscore'] = np.where(df['nutriscore']=='c', 3, df['nutriscore'])
df['nutriscore'] = np.where(df['nutriscore']=='d', 2, df['nutriscore'])
df['nutriscore'] = np.where(df['nutriscore']=='e', 1, df['nutriscore'])

for index, food in df.iterrows():
    utility_food = utility_func(df, index)
    preceding_foods = df[df['nutriscore'] == food['nutriscore']-1]
    for index2, preceding_food in preceding_foods.iterrows():
        utility_prec_food = utility_func(preceding_foods, index2)
        prob += (utility_food + sigma[index] >= utility_prec_food + sigma[index2] + 0.001)

prob += utility_thresh['energy'][len(utility_thresh['energy'])-1] == 0
prob += utility_thresh['saturated_fat'][len(utility_thresh['saturated_fat'])-1] == 0
prob += utility_thresh['sugars'][len(utility_thresh['sugars'])-1] == 0
prob += utility_thresh['salt'][len(utility_thresh['salt'])-1] == 0
prob += utility_thresh['proteins'][0] == 0
prob += utility_thresh['fiber'][0] == 0

prob += utility_thresh['energy'][0] + \
        utility_thresh['saturated_fat'][0] + \
        utility_thresh['sugars'][0] + \
        utility_thresh['salt'][0] + \
        utility_thresh['proteins'][len(utility_thresh['proteins'])-1] + \
        utility_thresh['fiber'][len(utility_thresh['fiber'])-1] == 1

for i in range(len(utility_thresh['energy'])-1):
    prob += utility_thresh['energy'][i] >= utility_thresh['energy'][i+1]
for i in range(len(utility_thresh['saturated_fat'])-1):
    prob += utility_thresh['saturated_fat'][i] >= utility_thresh['saturated_fat'][i+1]
for i in range(len(utility_thresh['sugars'])-1):
    prob += utility_thresh['sugars'][i] >= utility_thresh['sugars'][i+1]
for i in range(len(utility_thresh['salt'])-1):
    prob += utility_thresh['salt'][i] >= utility_thresh['salt'][i+1]

for i in range(len(utility_thresh['proteins'])-1):
    prob += utility_thresh['proteins'][i] <= utility_thresh['proteins'][i+1]
for i in range(len(utility_thresh['fiber'])-1):
    prob += utility_thresh['fiber'][i] <= utility_thresh['fiber'][i+1]

print('Solving the problem...')
prob.solve()
print("Status:", LpStatus[prob.status])

print('Saving marginal utility function plots...')

plt.rcParams["figure.figsize"] = [12,9]
plt.title('Utility function for criteria: Energy')

x = []

num_buckets = len(buckets['energy'])

for n in range(num_buckets):
  x.append(buckets['energy'][n][0])

x.append(buckets['energy'][num_buckets-1][1])

y = [p.value() for p in utility_thresh['energy']]

plt.plot(x,y)

plt.savefig('thresh_energy.jpg')
plt.clf()

plt.rcParams["figure.figsize"] = [12,9]
plt.title('Utility function for criteria: Saturated Fat')

x = []

num_buckets = len(buckets['saturated_fat'])

for n in range(num_buckets):
  x.append(buckets['saturated_fat'][n][0])

x.append(buckets['saturated_fat'][num_buckets-1][1])

y = [p.value() for p in utility_thresh['saturated_fat']]

plt.plot(x,y)

plt.savefig('thresh_saturated_fat.jpg')
plt.clf()

plt.rcParams["figure.figsize"] = [12,9]
plt.title('Utility function for criteria: Salt')

x = []

num_buckets = len(buckets['salt'])

for n in range(num_buckets):
  x.append(buckets['salt'][n][0])

x.append(buckets['salt'][num_buckets-1][1])

y = [p.value() for p in utility_thresh['salt']]

plt.plot(x,y)

plt.savefig('thresh_salt.jpg')
plt.clf()

plt.rcParams["figure.figsize"] = [12,9]
plt.title('Utility function for criteria: Sugars')

x = []

num_buckets = len(buckets['sugars'])

for n in range(num_buckets):
  x.append(buckets['sugars'][n][0])

x.append(buckets['sugars'][num_buckets-1][1])

y = [p.value() for p in utility_thresh['sugars']]

plt.plot(x,y)

plt.savefig('thresh_sugars.jpg')
plt.clf()

plt.rcParams["figure.figsize"] = [12,9]
plt.title('Utility function for criteria: Fiber')

x = []

num_buckets = len(buckets['fiber'])

for n in range(num_buckets):
  x.append(buckets['fiber'][n][0])

x.append(buckets['fiber'][num_buckets-1][1])

y = [p.value() for p in utility_thresh['fiber']]

plt.plot(x,y)

plt.savefig('thresh_fiber.jpg')
plt.clf()

plt.rcParams["figure.figsize"] = [12,9]
plt.title('Utility function for criteria: Protein')

x = []

num_buckets = len(buckets['proteins'])

for n in range(num_buckets):
  x.append(buckets['proteins'][n][0])

x.append(buckets['proteins'][num_buckets-1][1])

y = [p.value() for p in utility_thresh['proteins']]

plt.plot(x,y)

plt.savefig('thresh_proteins.jpg')
plt.clf()

print('Testing the model to generate preferences for 100 foods...')

utility_thresh_prod = {key: [lpVar.varValue for lpVar in utility_thresh[key]]for key in utility_thresh}

def utility_func_prod(df, food_index, utility_thresh):
    food = df.loc[food_index]
    criteria = list(food.keys())
    criteria.remove('nutriscore')
    utility = 0
    for criterion in criteria:
        bucket_index = get_bucket_index(food[criterion], buckets[criterion])
        left_thresh = buckets[criterion][bucket_index][0]
        right_thresh = buckets[criterion][bucket_index][1]
        m = (food[criterion] - left_thresh) / (right_thresh - food[criterion])
        left_utility = utility_thresh[criterion][bucket_index]
        right_utility = utility_thresh[criterion][bucket_index+1]
        
        utility += left_utility + m * (right_utility - left_utility)
    
    return utility

df_test_sampled = df_test.sample(100)

food_scores = []
for index, food in df_test_sampled.iterrows():
  pred = utility_func_prod(df_test_sampled, index, utility_thresh_prod)
  nutriscore = food['nutriscore']
  food_scores.append({'pred': pred, 'nutriscore': nutriscore})

food_scores.sort(key=lambda food: food['pred'], reverse=True)

with open('AdditiveNutriScoreResults.csv', 'w', newline='\n') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL, delimiter='\n')
    wr.writerow(food_scores)