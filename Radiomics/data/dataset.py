"""
Create a dataloader class from a dataframe, load selected columns as X and a column as Y. Add function to split into training, validation and test sets or stratified split or cross-validation split.
"""
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import MinMaxScaler


class Dataset():
    def __init__(self, dataframe, features, target, task_name=''):
        self.df = dataframe
        self.features = features
        self.target = target
        self.task_name = task_name
        self.X = self.df[self.features]
        self.y = self.df[self.target]
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.X_train_fold = None
        self.X_val_fold = None
        self.y_train_fold = None
        self.y_val_fold = None
        self.cv_split_generator = None
        self.scaler = MinMaxScaler()
        
    def split_dataset(self, test_size=0.2, val_size=0.2):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train, test_size=val_size)
        return self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test
    
    def stratified_split_dataset(self, test_size=0.2, val_size=0.2):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, stratify=self.y)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train, test_size=val_size, stratify=self.y_train)
        return self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test
    
    def cross_validation_split(self, n_splits=5, test_size=0.2):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, stratify=self.y)
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True)
        self.cv_split_generator = kf.split(self.X_train, self.y_train)
        return self.cv_split_generator
    
    def get_cross_validation_fold(self, train_index, val_index):
        if self.cv_split_generator == None:
            print(f"No folds found. Try running 'cross_validation_split' first.")
        else:
            self.X_train_fold, self.X_val_fold = self.X.iloc[train_index], self.X.iloc[val_index]
            self.y_train_fold, self.y_val_fold = self.y.iloc[train_index], self.y.iloc[val_index]
        return self.X_train_fold, self.X_val_fold, self.X_test, self.y_train_fold, self.y_val_fold, self.y_test
    
    def split_train_test_from_column(self, column_name, test_value):
        self.df_test = self.df[self.df[column_name] == test_value]
        self.df_train = self.df[self.df[column_name] != test_value]
        self.X_test = self.df_test[self.features]
        self.y_test = self.df_test[self.target]
        self.X_train = self.df_train[self.features]
        self.y_train = self.df_train[self.target]      
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def split_dataset_test_from_column(self, column_name, test_value, val_size=0.2):
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_train_test_from_column(column_name, test_value)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train, test_size=val_size)
        return self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test

    def cross_validation_split_test_from_column(self, column_name, test_value, n_splits=5):
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_train_test_from_column(column_name, test_value)
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True)
        self.cv_split_generator = kf.split(self.X_train, self.y_train)
        return self.cv_split_generator
 
    def standardize_features(self):
        self.X_train[self.X_train.columns] = self.scaler.fit_transform(self.X_train)
        if self.X_val is None:
            print(f'X_val not set. Leaving out.')
        else:
            self.X_val[self.X_val.columns] = self.scaler.transform(self.X_val)
        self.X_test[self.X_test.columns] = self.scaler.transform(self.X_test)

    def standardize_features_cross_validation(self):
        self.X_train_fold[self.X_train_fold.columns] = self.scaler.fit_transform(self.X_train_fold)
        self.X_val_fold[self.X_val_fold.columns] = self.scaler.transform(self.X_val_fold)

    def select_features(self):
        if self.X_train is None:
            raise ValueError('Split the data into training, validation and test first.')
        else:
            feature_selector = SelectKBest(f_classif, k=10)
            feature_selector.fit(self.X_train, self.y_train)
            cols = feature_selector.get_support(indices=True)
            self.X_train = self.X_train.iloc[:, cols]
            if self.X_val is not None:
                self.X_val = self.X_val.iloc[:, cols]
            self.X_test = self.X_test.iloc[:, cols]

    def select_features_cross_validation(self):
        if self.X_train is None:
            raise ValueError('Split the data into training and test first.')
        else:
            feature_selector = SelectKBest(f_classif, k=10)
            feature_selector.fit(self.X_train_fold, self.y_train_fold)
            cols = feature_selector.get_support(indices=True)
            self.X_train_fold = self.X_train_fold.iloc[:, cols]
            self.X_val_fold = self.X_val_fold.iloc[:, cols]

    def get_train_val_test(self):
        return self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test
    
    def get_X_y(self):
        return self.X, self.y
    
    def get_X_train_val_test(self):
        return self.X_train, self.X_val, self.X_test
    
    def get_y_train_val_test(self):
        return self.y_train, self.y_val, self.y_test
    
    def get_features(self):
        return self.features
    
    def get_target(self):
        return self.target
    
    def get_df(self):
        return self.df
    
    def get_X(self):
        return self.X
    
    def get_y(self):
        return self.y
    
    def get_X_train(self):
        return self.X_train
    
    def get_X_val(self):
        return self.X_val
    
    def get_X_test(self):
        return self.X_test
    
    def get_y_train(self):
        return self.y_train
    
    def get_y_val(self):
        return self.y_val
    
    def get_y_test(self):
        return self.y_test
    
    def get_X_y_train_val_test(self):
        return self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test
    
    def get_X_y_train_test(self):
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def get_X_y_train_val(self):
        return self.X_train, self.X_val, self.y_train, self.y_val
    
    def get_X_y_train(self):
        return self.X_train, self.y_train
    
    def get_X_y_val(self):
        return self.X_val, self.y_val
    
    def get_X_y_test(self):
        return self.X_test, self.y_test
    
    def get_X_y_train_val_test_names(self):
        return self.X_train.columns.values, self.X_val.columns.values, self.X_test.columns.values, self.y_train.name, self.y_val.name, self.y_test.name
    
    def get_X_y_train_test_names(self):
        return self.X_train.columns.values, self.X_test.columns.values, self.y_train.name, self.y_test.name
    
    def get_X_y_train_val_names(self):
        return self.X_train.columns.values, self.X_val.columns.values, self.y_train.name, self.y_val.name
    
    def get_X_y_train_names(self):
        return self.X_train.columns.values, self.y_train.name
    
    def get_X_y_val_names(self):
        return self.X_val.columns.values, self.y_val.name
    
    def get_X_y_test_names(self):
        return self.X_test.columns.values, self.y_test.name
    
    def get_X_y_train_val_test_df(self):
        return self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test
    
    def get_X_y_train_test_df(self):
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def get_X_y_train_val_df(self):
        return self.X_train, self.X_val, self.y_train, self.y_val
    
    def get_X_y_train_df(self):
        return self.X_train, self.y_train
    
    def get_X_y_val_df(self):
        return self.X_val, self.y_val
    
    def get_X_y_test_df(self):
        return self.X_test, self.y_test
    
    def get_X_y_train_val_test_df_names(self):
        return self.X_train.columns.values, self.X_val.columns.values, self.X_test.columns.values, self.y_train.name, self.y_val.name, self.y_test.name
    
    def get_X_y_train_test_df_names(self):
        return self.X_train.columns.values, self.X_test.columns.values, self.y_train.name, self.y_test.name
    
    def get_X_y_train_val_df_names(self):
        return self.X_train.columns.values, self.X_val.columns.values, self.y_train.name, self.y_val.name
    
    def get_X_y_train_df_names(self):
        return self.X_train.columns.values, self.y_train.name
    
    def get_X_y_val_df_names(self):
        return self.X_val.columns.values, self.y_val.name
    
    def get_X_y_test_df_names(self):
        return self.X_test.columns.values, self.y_test.name
    
   