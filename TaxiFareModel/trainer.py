# imports

from TaxiFareModel.data import clean_data, get_data, getXy, getholdout

from TaxiFareModel.utils import compute_rmse

from TaxiFareModel.encoders import DistanceTransformer,TimeFeaturesEncoder
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from sklearn.metrics import r2_score

class Trainer(BaseEstimator, TransformerMixin):
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y
    
    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        dist_pipe = Pipeline([
                        ('dist_trans', DistanceTransformer()),
                        ('stdscaler', StandardScaler())
                    ])
        time_pipe = Pipeline([
                        ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
                        ('ohe', OneHotEncoder(handle_unknown='ignore'))
                    ])
        preproc_pipe = ColumnTransformer([
                                        ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
                                        ('time', time_pipe, ['pickup_datetime'])
                                        ], remainder="drop")
        pipe = Pipeline([
                            ('preproc', preproc_pipe),
                            ('linear_model', LinearRegression())
                        ])
        return pipe        

    def run(self):
        """set and train the pipeline"""
        self.pipeline = self.set_pipeline()
        self.pipeline.fit(self.X, self.y)
        return self

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        return rmse
        
if __name__ == "__main__":
    # get data
    data = get_data(nrows=1000)
    # clean data
    data = clean_data(data)
    # set X and y
    X,y = getXy(data, col_target = "fare_amount")
    # hold out
    X_train, X_val, y_train, y_val = getholdout(X,y)
    # train
    trainer = Trainer(X_train,y_train)
    trainer.run()
    # evaluate
    score = trainer.evaluate(X_val,y_val)
    print(score)
