import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import k_fold_cross_validation
from lifelines import KaplanMeierFitter
from lifelines.utils import median_survival_times
from sksurv.linear_model import CoxPHSurvivalAnalysis

np.random.seed(123)

n = 500
X = pd.DataFrame({'X1': np.random.normal(size=n),
                  'X2': np.random.binomial(n=1, p=0.5, size=n)})


def S0(t, x):
    return np.exp(-np.exp(-2 + x['X1'] - x['X2'] + 0.5 * x['X1'] * x['X2']) * t)


T = np.random.exponential(scale=np.exp(-2 + X['X1'] - X['X2'] + 0.5 * X['X1'] * X['X2']), size=n)


def G0(t, x):
    return 0.9 * (t < 15) * np.exp(-np.exp(-2 - 0.5 * x['X1'] - 0.25 * x['X2'] + 0.5 * x['X1'] * x['X2']) * t)


C = np.random.exponential(scale=np.exp(-2 - 0.5 * X['X1'] - 0.25 * X['X2'] + 0.5 * X['X1'] * X['X2']), size=n)
C[C > 15] = 15

time = np.minimum(T, C)
event = np.array(T <= C).astype(int)

# SL_library = ['SL.mean', 'SL.glm', 'SL.gam', 'SL.earth']

fit = CoxPHFitter().fit(pd.DataFrame({'time': time, 'event': event, 'X1': X['X1'], 'X2': X['X2']}),
                        duration_col='time',
                        event_col='event',
                        formula='X1 + X2',
                        strata=['X2'],
                        robust=True,
                        show_progress=True)