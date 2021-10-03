

class Trainer():
    def __init__(self, dataset, model):
        self.dataset = dataset
        self.model = model
        self.param_grid = None

    def get_grid_RandomForest(self):
        n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
        max_features = ['auto', 'sqrt']
        max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
        max_depth.append(None)
        min_samples_split = [2, 5, 10]
        min_samples_leaf = [1, 2, 4]
        bootstrap = [True, False]

        self.param_grid = {'n_estimators': n_estimators,
                           'max_features': max_features,
                           'max_depth': max_depth,
                           'min_samples_split': min_samples_split,
                           'min_samples_leaf': min_samples_leaf,
                           'bootstrap': bootstrap}
        return self
    
    def get_grid_XGBoost(self):
        self.param_grid = {"learning_rate": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
                           "max_depth": [3, 4, 5, 6, 8, 10, 12, 15],
                           "min_child_weight": [1, 3, 5, 7],
                           "gamma": [0.0, 0.1, 0.2, 0.3, 0.4],
                           "colsample_bytree": [0.3, 0.4, 0.5, 0.7]}
        return self
    
    def get_grid_LogReg(self)
        self.param_grid = {"C": [0.001, 0.01, 0.1, 1, 10, 100, 1000], 
                           "penalty": ["l1", "l2", "elasticnet", "none"]}
        return self

    def update_model_params_random_search(self)
        if self.param_grid is None:
            raise ValueError('First select param grid!')
        else:
            param_searcher = RandomizedSearchCV(estimator=self.model, param_distributions=self.param_grid,
                                    n_iter=400, cv=5, verbose=2,
                                    random_state=42, n_jobs=-1)
            rs = param_searcher.fit(self.dataset.X_train, self.dataset.y_train)
            self.model = rs.best_estimator_
            
            return self
