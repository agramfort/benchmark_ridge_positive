from benchopt import BaseDataset, safe_import_context


with safe_import_context() as import_ctx:
    import numpy as np
    from scipy import sparse


class Dataset(BaseDataset):

    name = "Simulated"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    parameters = {
        "n_samples, n_features, use_sparse": [
            (100, 1000, False),
            (500, 2000, True),
        ],
    }

    def __init__(self, n_samples=10, n_features=50, use_sparse=False,
                 random_state=27):
        # Store the parameters of the dataset
        self.n_samples = n_samples
        self.n_features = n_features
        self.use_sparse = use_sparse
        self.random_state = random_state

    def get_data(self):

        rng = np.random.RandomState(self.random_state)
        w = rng.randn(self.n_features)
        w[w < 0] = 0
        if self.use_sparse:
            X = rng.randn(self.n_samples, self.n_features)
        else:
            X = sparse.rand(
                self.n_samples,
                self.n_features,
                density=0.4,
                format="csc",
                dtype=np.float,
                random_state=rng,
            )
        y = X @ w
        y += 0.1 * y.std() * rng.randn(self.n_samples)

        data = dict(X=X, y=y)

        return self.n_features, data
