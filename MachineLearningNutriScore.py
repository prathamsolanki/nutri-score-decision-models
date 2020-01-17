import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn import tree
import graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix

print('Importing the dataset...')
raw_df = pd.read_excel('openfoodfacts_simplified_database.xlsx')
raw_df = raw_df.set_index('product_name')

raw_df = raw_df.replace(['a','b','c','d','e'], [0,1,2,3,4])

features = ['energy_100g','saturated-fat_100g','sugars_100g','fiber_100g','proteins_100g','sodium_100g']
label = ['nutrition_grade_fr']

raw_df = raw_df[features + label]

raw_df = raw_df.dropna()

X = raw_df[features]

y = raw_df[label]

print('Splitting the dataset...')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)

X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=1)

X_train.count()[0], X_val.count()[0], X_test.count()[0]

print('Training a Decision Tree...')

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

dot_data = tree.export_graphviz(clf, out_file=None, 
                      feature_names=features,  
                      class_names='nutrition_grade_fr',  
                      filled=True, rounded=True,  
                      special_characters=True)  
graph = graphviz.Source(dot_data)  
graph.render("nutri-score")

print('Accuracy: ', clf.score(X_test,y_test))

plt.rcParams["figure.figsize"] = [12,9]

# Plot non-normalized confusion matrix
titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]

for title, normalize in titles_options:
    disp = plot_confusion_matrix(clf, X_test, y_test,
                                 display_labels=['A','B','C','D','E'],
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)

    # print(title)
    # print(disp.confusion_matrix)
    if normalize is None:
      plt.savefig('confusion_matrix_dt.png')
    else:
      plt.savefig('confusion_matrix_norm_dt.png')

print('Training a Random Forest...')

clf = RandomForestClassifier(random_state=0)
clf.fit(X_train, y_train)

print('Accuracy: ', clf.score(X_test,y_test))

plt.rcParams["figure.figsize"] = [12,9]

# Plot non-normalized confusion matrix
titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]

for title, normalize in titles_options:
    disp = plot_confusion_matrix(clf, X_test, y_test,
                                 display_labels=['A','B','C','D','E'],
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)

    # print(title)
    # print(disp.confusion_matrix)
    if normalize is None:
      plt.savefig('confusion_matrix_rf.png')
    else:
      plt.savefig('confusion_matrix_norm_rf.png')

print('Training XGBoost...')

def runXGB(train_X, train_y, validation_X, validation_y, test_X):
    param = {}
    param['objective'] = 'multi:softmax'
    param['num_class'] = 5
    param['eta'] = 0.01
    param['max_depth'] = 6
    param['gamma'] = 0
    param['eval_metric'] = "mlogloss"
    param['min_child_weight'] = 1
    param['max_delta_step'] = 0
    param['subsample'] = 1
    param['colsample_bytree'] = 1
    param['lambda'] = 1
    param['alpha'] = 0
    param['seed'] = 0
    param['verbosity'] = 0
    num_rounds = 10000

    plst = list(param.items())

    xgtrain = xgb.DMatrix(train_X, label = train_y)
    xgcv = xgb.DMatrix(validation_X, label = validation_y)
    xgtest = xgb.DMatrix(test_X)

    evallist = [(xgcv,'eval')]
    model = xgb.train(plst, xgtrain, num_rounds, evallist, early_stopping_rounds = 100)

    pred_test_y = model.predict(xgtest)
    return pred_test_y, model

preds, model = runXGB(X_train, y_train, X_val, y_val, X_test)

print('Accuracy: ', (y_test['nutrition_grade_fr'] == preds).mean())

plt.rcParams["figure.figsize"] = [12,9]

cm = confusion_matrix(y_test['nutrition_grade_fr'], preds, labels=[0,1,2,3,4])

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
plt.savefig('confusion_matrix_xgb.png')

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
plt.savefig('confusion_matrix_norm_xgb.png')