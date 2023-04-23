# pip install pysurvival
from pysurvival.models.survival_forest import RandomSurvivalForestModel
#### 1 - Importing packages
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from pysurvival.models.simulations import SimulationModel
from pysurvival.models.survival_forest import ConditionalSurvivalForestModel
from pysurvival.utils.metrics import concordance_index
from pysurvival.utils.display import integrated_brier_score
%pylab inline

#### 2 - Generating the dataset from a Exponential parametric model
# Initializing the simulation model
sim = SimulationModel( survival_distribution = 'exponential',
                       risk_type = 'linear',
                       censored_parameter = 1,
                       alpha = 3)

# Generating N random samples
N = 1000
dataset = sim.generate_data(num_samples = N, num_features=4)

# Showing a few data-points
dataset.head(2)

from pysurvival.utils.display import display_baseline_simulations
display_baseline_simulations(sim, figure_size=(20, 6))

#### 3 - Creating the modeling dataset
# Defining the features
features = sim.features

# Building training and testing sets #
index_train, index_test = train_test_split( range(N), test_size = 0.2)
data_train = dataset.loc[index_train].reset_index( drop = True )
data_test  = dataset.loc[index_test].reset_index( drop = True )

# Creating the X, T and E input
X_train, X_test = data_train[features], data_test[features]
T_train, T_test = data_train['time'].values, data_test['time'].values
E_train, E_test = data_train['event'].values, data_test['event'].values


#### 4 - Creating an instance of the Conditional model and fitting the data.
# Building the model
csf = ConditionalSurvivalForestModel(num_trees=200)
csf.fit(X_train, T_train, E_train,
        max_features="sqrt", max_depth=5, min_node_size=20,
        alpha = 0.05, minprop=0.1)


#### 5 - Cross Validation / Model Performances
c_index = concordance_index(csf, X_test, T_test, E_test) #0.81
print('C-index: {:.2f}'.format(c_index))

ibs = integrated_brier_score(csf, X_test, T_test, E_test, t_max=30,
            figure_size=(20, 6.5) )
print('IBS: {:.2f}'.format(ibs))


# Initializing the figure
fig, ax = plt.subplots(figsize=(8, 4))

# Randomly extracting a data-point that experienced an event
choices = np.argwhere((E_test==1.)&(T_test>=1)).flatten()
k = np.random.choice( choices, 1)[0]

# Saving the time of event
t = T_test[k]

# Computing the Survival function for all times t
survival = csf.predict_survival(X_test.values[k, :]).flatten()
actual = sim.predict_survival(X_test.values[k, :]).flatten()

# Displaying the functions
plt.plot(csf.times, survival, color = 'blue', label='predicted', lw=4, ls = '-.')
plt.plot(sim.times, actual, color = 'red', label='actual', lw=2)

# Actual time
plt.axvline(x=t, color='black', ls ='--')
ax.annotate('T={:.1f}'.format(t), xy=(t, 0.5), xytext=(t, 0.5), fontsize=12)

# Show everything
title = "Comparing Survival functions between Actual and Predicted"
plt.legend(fontsize=12)
plt.title(title, fontsize=15)
plt.ylim(0, 1.05)
plt.show()

