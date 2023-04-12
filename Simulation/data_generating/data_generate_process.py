from Simulation.data_generating.DGP_COX import SimulationModel

sim = SimulationModel( survival_distribution='exponential',
                       risk_type='linear',
                       alpha=1,
                       beta=1
                       )


def data_generate(N):
    dataset = sim.generate_data(num_samples = N, num_features=2,
                            feature_weights=[-1,1],
                            treatment_weights=[2])

    dataset.columns = ['x', 'a', 'o', 'e']

    return dataset


# df = data_generate(N=100)