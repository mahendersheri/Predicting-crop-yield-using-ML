# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 13:57:29 2020

@author: MAHENDER
"""

# Pandas is used for data manipulation
import pandas as pd
# Read in data and display first 5 rows
features = pd.read_csv('E:\prediction\prd\ricenorm.csv')
features.head(5)

print('The shape of our features is:', features.shape)

features.describe()


# Use numpy to convert to arrays
import numpy as np
# PRODUCTION are the values we want to predict
labels = np.array(features['PRODUCTION'])
# Remove the labels from the features
# axis 1 refers to the columns
features= features.drop('PRODUCTION', axis = 1)
# Saving feature names for later use
feature_list = list(features.columns)
# Convert to numpy array
features = np.array(features)


# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)

print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)


# The baseline predictions are the historical averages
#baseline_preds = test_features[:, feature_list.index('average')]
# Baseline errors, and display average baseline error
#baseline_errors = abs(baseline_preds - test_labels)
#print('Average baseline error: ', round(np.mean(baseline_errors), 2))

# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
# Train the model on training data
rf.fit(train_features, train_labels);

# Use the forest's predict method on the test data
predictions = rf.predict(test_features)
# Calculate the absolute errors
errors = abs(predictions - test_labels)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')


# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / test_labels)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')

# Import tools needed for visualization
from sklearn.tree import export_graphviz
import pydot
# Pull out one tree from the forest
tree = rf.estimators_[5]
# Import tools needed for visualization
from sklearn.tree import export_graphviz
import pydot
# Pull out one tree from the forest
tree = rf.estimators_[5]
# Export the image to a dot file
export_graphviz(tree, out_file = 'treeRice.dot', feature_names = feature_list, rounded = True, precision = 1)
# Use dot file to create a grapho
(graph, ) = pydot.graph_from_dot_file('treeRice.dot')
# Write graph to a png file
graph.write_png('treeRice.png')

# Limit depth of tree to 3 levels
rf_small = RandomForestRegressor(n_estimators=10, max_depth = 3)
rf_small.fit(train_features, train_labels)
# Extract the small tree
tree_small = rf_small.estimators_[5]
# Save the tree as a png image
export_graphviz(tree_small, out_file = 'small_treeRice.dot', feature_names = feature_list, rounded = True, precision = 1)
(graph, ) = pydot.graph_from_dot_file('small_treeRice.dot')
graph.write_png('small_treeRice.png');

# Get numerical feature importances
importances = list(rf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];



"""
# Make the data accessible for plotting
true_data['GROUND_WARELEVEL'] = features[:, feature_list.index('GROUND_WATERLEVEL')]
true_data['RAINFALL'] = features[:, feature_list.index('RAINFALL')]
true_data['AREA'] = features[:, feature_list.index('AREA')]
# Plot all the data as lines
plt.plot(true_data['PRODUCTION'], true_data['GROUND_WATERLEVEL'], 'b-', label  = 'GROUND_WATERLEVEL', alpha = 1.0)
plt.plot(true_data['PRODUCTION'], true_data['AREA'], 'y-', label  = 'AREA', alpha = 1.0)
plt.plot(true_data['PRODUCTION'], true_data['RAINFALL'], 'k-', label = 'RAINFALL', alpha = 0.8)
plt.plot(true_data['PRODUCTION'], true_data['AVG_TEMPERATURE'], 'r-', label = 'AVG_TEMPERARURE', alpha = 0.3)
# Formatting plot
plt.legend(); plt.xticks(rotation = '60');
# Lables and title
plt.xlabel('PRODUCTION'); plt.ylabel('PRODUCTION'); plt.title('VARIABLE ');
"""