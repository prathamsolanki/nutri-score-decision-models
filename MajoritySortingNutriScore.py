import warnings
warnings.filterwarnings("ignore")

from pulp import *
import pandas as pd
import sys
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

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

max_energy = df['energy'].max()
min_energy = df['energy'].min()

max_saturated_fat = df['saturated_fat'].max()
min_saturated_fat = df['saturated_fat'].min()

max_sugars = df['sugars'].max()
min_sugars = df['sugars'].min()

max_fiber = df['fiber'].max()
min_fiber = df['fiber'].min()

max_proteins = df['proteins'].max()
min_proteins = df['proteins'].min()

max_salt = df['salt'].max()
min_salt = df['salt'].min()

df['nutriscore'] = np.where(df['nutriscore']=='a', 5, df['nutriscore'])
df['nutriscore'] = np.where(df['nutriscore']=='b', 4, df['nutriscore'])
df['nutriscore'] = np.where(df['nutriscore']=='c', 3, df['nutriscore'])
df['nutriscore'] = np.where(df['nutriscore']=='d', 2, df['nutriscore'])
df['nutriscore'] = np.where(df['nutriscore']=='e', 1, df['nutriscore'])

# Normalize the energy column
df['energy'] = (df['energy']-min_energy) / (max_energy-min_energy)*100

print('Splitting the dataset...')

features = ['energy', 'proteins', 'salt', 'fiber', 'saturated_fat', 'sugars']
X = df[features]
y = df[['nutriscore']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

df = X_train
df['nutriscore'] = y_train['nutriscore']

df.count()[0]

test_df = X_test
test_df['nutriscore'] = y_test['nutriscore']

test_df.count()[0]

print('Formalizing the problem to learn the limiting profiles...')

prob = LpProblem("The Nutriscore", LpMinimize)

pi = []
pi.append([
           LpVariable('pi_1_energy', 0, 100),
           LpVariable('pi_1_saturated_fat', 0, 100),
           LpVariable('pi_1_sugars', 0, 100),
           LpVariable('pi_1_fiber', 0, 100),
           LpVariable('pi_1_proteins', 0, 100),
           LpVariable('pi_1_salt', 0, 100),
           ])
pi.append([
           LpVariable('pi_2_energy', 0, 100),
           LpVariable('pi_2_saturated_fat', 0, 100),
           LpVariable('pi_2_sugars', 0, 100),
           LpVariable('pi_2_fiber', 0, 100),
           LpVariable('pi_2_proteins', 0, 100),
           LpVariable('pi_2_salt', 0, 100),
           ])
pi.append([
           LpVariable('pi_3_energy', 0, 100),
           LpVariable('pi_3_saturated_fat', 0, 100),
           LpVariable('pi_3_sugars', 0, 100),
           LpVariable('pi_3_fiber', 0, 100),
           LpVariable('pi_3_proteins', 0, 100),
           LpVariable('pi_3_salt', 0, 100),
           ])
pi.append([
           LpVariable('pi_4_energy', 0, 100),
           LpVariable('pi_4_saturated_fat', 0, 100),
           LpVariable('pi_4_sugars', 0, 100),
           LpVariable('pi_4_fiber', 0, 100),
           LpVariable('pi_4_proteins', 0, 100),
           LpVariable('pi_4_salt', 0, 100),
           ])
pi.append([
           LpVariable('pi_5_energy', 0, 100),
           LpVariable('pi_5_saturated_fat', 0, 100),
           LpVariable('pi_5_sugars', 0, 100),
           LpVariable('pi_5_fiber', 0, 100),
           LpVariable('pi_5_proteins', 0, 100),
           LpVariable('pi_5_salt', 0, 100),
           ])
pi.append([
           LpVariable('pi_6_energy', 0, 100),
           LpVariable('pi_6_saturated_fat', 0, 100),
           LpVariable('pi_6_sugars', 0, 100),
           LpVariable('pi_6_fiber', 0, 100),
           LpVariable('pi_6_proteins', 0, 100),
           LpVariable('pi_6_salt', 0, 100),
           ])

# Error to minimize
eps = {}

for index, food in df.iterrows():
  eps[index] = []

  eps[index].append([
            LpVariable('eps_' + str(index) + '_1_energy', 0, 100),
            LpVariable('eps_' + str(index) + '_1_saturated_fat', 0, 100),
            LpVariable('eps_' + str(index) + '_1_sugars', 0, 100),
            LpVariable('eps_' + str(index) + '_1_fiber', 0, 100),
            LpVariable('eps_' + str(index) + '_1_proteins', 0, 100),
            LpVariable('eps_' + str(index) + '_1_salt', 0, 100),
            ])
  eps[index].append([
            LpVariable('eps_' + str(index) + '_2_energy', 0, 100),
            LpVariable('eps_' + str(index) + '_2_saturated_fat', 0, 100),
            LpVariable('eps_' + str(index) + '_2_sugars', 0, 100),
            LpVariable('eps_' + str(index) + '_2_fiber', 0, 100),
            LpVariable('eps_' + str(index) + '_2_proteins', 0, 100),
            LpVariable('eps_' + str(index) + '_2_salt', 0, 100),
            ])
  eps[index].append([
            LpVariable('eps_' + str(index) + '_3_energy', 0, 100),
            LpVariable('eps_' + str(index) + '_3_saturated_fat', 0, 100),
            LpVariable('eps_' + str(index) + '_3_sugars', 0, 100),
            LpVariable('eps_' + str(index) + '_3_fiber', 0, 100),
            LpVariable('eps_' + str(index) + '_3_proteins', 0, 100),
            LpVariable('eps_' + str(index) + '_3_salt', 0, 100),
            ])
  eps[index].append([
            LpVariable('eps_' + str(index) + '_4_energy', 0, 100),
            LpVariable('eps_' + str(index) + '_4_saturated_fat', 0, 100),
            LpVariable('eps_' + str(index) + '_4_sugars', 0, 100),
            LpVariable('eps_' + str(index) + '_4_fiber', 0, 100),
            LpVariable('eps_' + str(index) + '_4_proteins', 0, 100),
            LpVariable('eps_' + str(index) + '_4_salt', 0, 100),
            ])
  eps[index].append([
            LpVariable('eps_' + str(index) + '_5_energy', 0, 100),
            LpVariable('eps_' + str(index) + '_5_saturated_fat', 0, 100),
            LpVariable('eps_' + str(index) + '_5_sugars', 0, 100),
            LpVariable('eps_' + str(index) + '_5_fiber', 0, 100),
            LpVariable('eps_' + str(index) + '_5_proteins', 0, 100),
            LpVariable('eps_' + str(index) + '_5_salt', 0, 100),
            ])
  eps[index].append([
            LpVariable('eps_' + str(index) + '_6_energy', 0, 100),
            LpVariable('eps_' + str(index) + '_6_saturated_fat', 0, 100),
            LpVariable('eps_' + str(index) + '_6_sugars', 0, 100),
            LpVariable('eps_' + str(index) + '_6_fiber', 0, 100),
            LpVariable('eps_' + str(index) + '_6_proteins', 0, 100),
            LpVariable('eps_' + str(index) + '_6_salt', 0, 100),
            ])

eps_flattened = list(itertools.chain.from_iterable(list(itertools.chain.from_iterable(eps.values()))))

prob += lpSum(eps_flattened), "Minimize the error in the thresholds"

for index, food in df.iterrows():
  score = food['nutriscore']
  pi_lower = score-1
  pi_upper = score
  prob += (food['energy'] <= pi[pi_lower][0] + eps[index][pi_lower][0])
  prob += (food['energy'] >= pi[pi_upper][0] - eps[index][pi_upper][0])
  prob += (food['saturated_fat'] <= pi[pi_lower][1] + eps[index][pi_lower][1])
  prob += (food['saturated_fat'] >= pi[pi_upper][1] - eps[index][pi_upper][1])
  prob += (food['sugars'] <= pi[pi_lower][2] + eps[index][pi_lower][2])
  prob += (food['sugars'] >= pi[pi_upper][2] - eps[index][pi_upper][2])
  prob += (food['fiber'] >= pi[pi_lower][3] - eps[index][pi_lower][3])
  prob += (food['fiber'] <= pi[pi_upper][3] + eps[index][pi_upper][3])
  prob += (food['proteins'] >= pi[pi_lower][4] - eps[index][pi_lower][4])
  prob += (food['proteins'] <= pi[pi_upper][4] + eps[index][pi_upper][4])
  prob += (food['salt'] <= pi[pi_lower][5] + eps[index][pi_lower][5])
  prob += (food['salt'] >= pi[pi_upper][5] - eps[index][pi_upper][5])

prob += (pi[0][0] == 100)
prob += (pi[0][1] == 100)
prob += (pi[0][2] == 100)
prob += (pi[0][3] == 0)
prob += (pi[0][4] == 0)
prob += (pi[0][5] == 100)

prob += (pi[5][0] == 0)
prob += (pi[5][1] == 0)
prob += (pi[5][2] == 0)
prob += (pi[5][3] == 100)
prob += (pi[5][4] == 100)
prob += (pi[5][5] == 0)

prob += (pi[0][0] >= pi[1][0]+1)
prob += (pi[1][0] >= pi[2][0]+1)
prob += (pi[2][0] >= pi[3][0]+1)
prob += (pi[3][0] >= pi[4][0]+1)
prob += (pi[4][0] >= pi[5][0]+1)

prob += (pi[0][1] >= pi[1][1]+1)
prob += (pi[1][1] >= pi[2][1]+1)
prob += (pi[2][1] >= pi[3][1]+1)
prob += (pi[3][1] >= pi[4][1]+1)
prob += (pi[4][1] >= pi[5][1]+1)

prob += (pi[0][2] >= pi[1][2]+1)
prob += (pi[1][2] >= pi[2][2]+1)
prob += (pi[2][2] >= pi[3][2]+1)
prob += (pi[3][2] >= pi[4][2]+1)
prob += (pi[4][2] >= pi[5][2]+1)

prob += (pi[0][3] <= pi[1][3]-1)
prob += (pi[1][3] <= pi[2][3]-1)
prob += (pi[2][3] <= pi[3][3]-1)
prob += (pi[3][3] <= pi[4][3]-1)
prob += (pi[4][3] <= pi[5][3]-1)

prob += (pi[0][4] <= pi[1][4]-1)
prob += (pi[1][4] <= pi[2][4]-1)
prob += (pi[2][4] <= pi[3][4]-1)
prob += (pi[3][4] <= pi[4][4]-1)
prob += (pi[4][4] <= pi[5][4]-1)

prob += (pi[0][5] >= pi[1][5]+1)
prob += (pi[1][5] >= pi[2][5]+1)
prob += (pi[2][5] >= pi[3][5]+1)
prob += (pi[3][5] >= pi[4][5]+1)
prob += (pi[4][5] >= pi[5][5]+1)

print('Solving the problem to learn the limiting profiles...')
prob.solve()
print("Status:", LpStatus[prob.status])

pi_lp = pi
pi = [[var.value() for var in p] for p in pi_lp]

pi = np.array(pi)

# rescaling the energy
pi[:,0] = pi[:,0]/100 * (max_energy-min_energy)

pi

print('Testing MR-Sort using the learnt profiles...')

test_df['energy'] = test_df['energy']/100 * (max_energy-min_energy)

def predict(food, pi):
  w = 1/6
  thresh = 0.5
  starting_index = 0
  ending_index = len(pi)-1
  
  max_score = 0
  max_index = starting_index
  
  i = starting_index
  while (i < ending_index):
    index = i
    adj_index = index+1

    score = 0
    if (food['energy'] >= pi[adj_index][0] and food['energy'] <= pi[index][0]):
        score += w
    if (food['saturated_fat'] >= pi[adj_index][1] and food['saturated_fat'] <= pi[index][1]):
        score += w
    if (food['sugars'] >= pi[adj_index][2] and food['sugars'] <= pi[index][2]):
        score += w
    if (food['fiber'] <= pi[adj_index][3] and food['fiber'] >= pi[index][3]):
        score += w
    if (food['proteins'] <= pi[adj_index][4] and food['proteins'] >= pi[index][4]):
        score += w
    if (food['salt'] >= pi[adj_index][5] and food['salt'] <= pi[index][5]):
        score += w
      
    if score > max_score:
      max_score = score
      max_index = index
    i += 1
  
  return max_index+1

y_pred = []

for index, food in test_df.iterrows():
  pred = predict(food, pi)
  y_pred.append(pred)

print('Accuracy: ', (test_df['nutriscore'] == y_pred).mean())

plt.rcParams["figure.figsize"] = [12,9]

cm = confusion_matrix(list(test_df['nutriscore']), y_pred, labels=[1,2,3,4,5])

labels = ['A','B','C','D','E']

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confusion matrix, without normalization')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.savefig('confusion_matrix_mrsort_auto.jpg')

cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm_norm)
plt.title('Normalized confusion matrix')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.savefig('confusion_matrix_norm_mrsort_auto.jpg')

print('Testing MR-Sort using the manual profiles...')

pi_2 = np.array([
                 [7510, 100, 100, 0, 0, 100],
                 [2009.9287999838757, 13.665507455827129, 33.394564151406406, 4.6768, 13.762, 2],
                 [1663.6701553788262, 7.0429976703970345, 22.053303416636737, 5.252, 18.3499, 1.2891238778338736],
                 [1446, 2.491991002779214, 13.354606049373178, 5.3, 19, 0.546764428138622],
                 [1205.146460873095, 1.1531811366411016, 8.106302920220942, 5.4, 20, 0.3882400255443486],
                 [0, 0, 0, 100, 100, 0]
])

# pi_2 = np.array([
#                  [7510, 100, 100, 0, 0, 100],
#                  [3698, 42, 100, 41, 41, 8],
#                  [3464, 60, 71, 27.0, 29, 19.52],
#                  [2920, 8.9, 67, 12.8, 27, 2.64],
#                  [2575, 6.2, 37, 47.8, 32, 5.2],
#                  [0, 0, 0, 100, 100, 0]
# ])

# pi_3 = np.array([
#                  [7510, 100, 100, 0, 0, 100],
#                  [2009.9287999838757, 13.665507455827129, 33.394564151406406, 11, 12.5, 4],
#                  [1663.6701553788262, 7.0429976703970345, 22.053303416636737, 12.5, 13.5, 3],
#                  [1446, 2.491991002779214, 13.354606049373178, 13.5, 14.5, 2],
#                  [1205.146460873095, 1.1531811366411016, 8.106302920220942, 14.5, 15.5, 1],
#                  [0, 0, 0, 100, 100, 0]
# ])

# pi_3 = np.array([
#                  [7510, 100, 100, 0, 0, 100],
#                  [1907, 21, 50, 4.6768, 13.762, 2],
#                  [1732, 20, 35.5, 5.252, 18.3499, 1.2891238778338736],
#                  [1460, 4.45, 33.5, 5.3, 19, 0.546764428138622],
#                  [1287, 3.1, 18.5, 5.4, 20, 0.3882400255443486],
#                  [0, 0, 0, 100, 100, 0]
# ])

y_pred = []

for index, food in test_df.iterrows():
  pred = predict(food, pi_2)
  y_pred.append(pred)

print('Accuracy: ', (test_df['nutriscore'] == y_pred).mean())

plt.rcParams["figure.figsize"] = [12,9]

cm = confusion_matrix(list(test_df['nutriscore']), y_pred, labels=[1,2,3,4,5])

labels = ['A','B','C','D','E']

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confusion matrix, without normalization')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.savefig('confusion_matrix_mrsort_manual.jpg')

cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm_norm)
plt.title('Normalized confusion matrix')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.savefig('confusion_matrix_norm_mrsort_manual.jpg')