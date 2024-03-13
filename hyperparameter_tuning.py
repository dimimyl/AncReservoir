from loadDataset import loadDataset
from reservoirpy.nodes import Reservoir, Ridge
from reservoirpy.observables import nrmse, rsquare
import numpy as np
import json
from reservoirpy.hyper import research

def objective(dataset, config, *, iss, N, sr, lr, ridge, seed):

    x_train, y_train, x_test, y_test= dataset
    # You can access anything you put in the config
    # file from the 'config' parameter.
    instances = config["instances_per_trial"]

    # The seed should be changed across the instances,
    # to be sure there is no bias in the results
    # due to initialization.
    variable_seed = seed

    losses = []
    r2s = []
    for n in range(instances):
        # Build your model given the input parameters
        reservoir = Reservoir(N,
                              sr=sr,
                              lr=lr,
                              input_scaling=iss,
                              seed=variable_seed)

        readout = Ridge(ridge=ridge)

        model = reservoir >> readout

        predictions = model.fit(x_train, y_train) \
                           .run(x_test)

        loss = nrmse(y_test, predictions, norm_value=np.ptp(x_train))
        r2 = rsquare(y_test, predictions)

        # Change the seed between instances
        variable_seed += 1

        losses.append(loss)
        r2s.append(r2)

    # Return a dictionnary of metrics. The 'loss' key is mandatory when
    # using hyperopt.
    return {'loss': np.mean(losses),
            'r2': np.mean(r2s)}

def hyper_tuning(dataset):
    hyperopt_config = {
        "exp": f"config1",  # the experimentation name
        "hp_max_evals": 200,  # the number of differents sets of parameters hyperopt has to try
        "hp_method": "random",  # the method used by hyperopt to chose those sets (see below)
        "seed": 42,  # the random state seed, to ensure reproducibility
        "instances_per_trial": 3,  # how many random ESN will be tried with each sets of parameters
        "hp_space": {  # what are the ranges of parameters explored
            "N": ["choice", 500],  # the number of neurons is fixed to 300
            "sr": ["loguniform", 1e-2, 10],  # the spectral radius is log-uniformly distributed between 1e-6 and 10
            "lr": ["loguniform", 1e-3, 1],  # idem with the leaking rate, from 1e-3 to 1
            "iss": ["choice", 0.9],  # the input scaling is fixed
            "ridge": ["choice", 1e-7],  # and so is the regularization parameter.
            "seed": ["choice", 1234]  # an other random seed for the ESN initialization
        }
    }

    import json

    # we precautionously save the configuration in a JSON file
    # each file will begin with a number corresponding to the current experimentation run number.
    with open(f"{hyperopt_config['exp']}.config.json", "w+") as f:
        json.dump(hyperopt_config, f)

    best = research(objective, dataset, f"/home/dimimyl/PycharmProjects/AncReservoir/{hyperopt_config['exp']}.config.json", ".")
