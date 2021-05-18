# imports
from TaxiFareModel.data import clean_data, get_data, getXy, getholdout
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.encoders import DistanceTransformer,TimeFeaturesEncoder
import mlflow
from memoized_property import memoized_property
from  mlflow.tracking import MlflowClient
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression,Lasso
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

class Trainer(BaseEstimator, TransformerMixin):
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y
        self.experiment_name = "[FR] [Paris] [MartinJ9678] taxifare_model 0"
        self.best_model=None
        self.best_params=None
    
    @memoized_property
    def mlflow_client(self):
        MLFLOW_URI = "https://mlflow.lewagon.co/"
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()
    
    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    def mlflow_run(self):
        self.mlflow_run_object = self.mlflow_client.create_run(self.mlflow_experiment_id)
        return self.mlflow_run_object

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run_object.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run_object.info.run_id, key, value)
    
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
                            ('linear_model', RandomForestRegressor(n_jobs=-1))
                        ])
        return pipe        

    def run(self):
        """set and train the pipeline"""
        self.pipeline = self.set_pipeline()
        grid={
            'linear_model__max_depth':[2,3],
            'linear_model__min_samples_leaf': [1,2],
            'linear_model__n_estimators': [100,150,200]
        }
        grid_search = GridSearchCV(self.pipeline,param_grid=grid,n_jobs=-1,cv=5,scoring='neg_mean_squared_error')
        grid_search.fit(self.X,self.y)
        self.best_model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        #self.pipeline.fit(self.X, self.y)
        return self

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.best_model.predict(X_test)
        #y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        # for model in ["a", "b"]: on peut itérer sur des modèles avec des runs différents en bouclant 
        #     self.mlflow_run()
        #     self.mlflow_log_metric("rmse",rmse)
        #     self.mlflow_log_param("model",self.pipeline.get_params()['linear_model'])
        #     self.mlflow_log_param("truc",model)
        #     for key,value in self.best_params.items():
        #         self.mlflow_log_param(key,value)
        self.mlflow_run()
        self.mlflow_log_metric("rmse",rmse)
        self.mlflow_log_param("model",self.pipeline.get_params()['linear_model'])
        for key,value in self.best_params.items():
            self.mlflow_log_param(key,value)
        return rmse
    
    def save_model(self,score,model):
        """ Save the trained model into a model.joblib file """
        joblib.dump(self.pipeline, f'model.joblib--{model}-{score}')
        return self
        
if __name__ == "__main__":
    # get data
    data = get_data(nrows=10000)
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
    # joblib
    model=trainer.pipeline.get_params()['linear_model']
    trainer.save_model(score,model)
    print(trainer.best_params)
