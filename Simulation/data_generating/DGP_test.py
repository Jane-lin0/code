from Simulation.data_generating.DGP_pysurvival import SimulationModel


sim = SimulationModel( survival_distribution='exponential',
                       risk_type='linear',
                       alpha=1,
                       beta=1
                       )
# sim = SimulationModel( survival_distribution = 'gompertz',
#                        risk_type = 'linear',
#                        censored_parameter = 5.0,
#                        alpha = 0.01,
#                        beta = 5., )

# Generating N Random samples
N = 1000
# dataset = sim.generate_data(num_samples = N,
#                             num_features=5,
#                             feature_weights=[1,2,3,-1,1],
#                             treatment_weights=[2,1,3,-2]) # T 很小，在0-2之间

# dataset = sim.generate_data(num_samples = N,
#                             num_features=3,
#                             feature_weights=[2,-1,1],
#                             treatment_weights=[-1,2])

dataset = sim.generate_data(num_samples = N,
                            num_features=2,
                            feature_weights=[-1,1],
                            treatment_weights=[2])
print(dataset.describe())