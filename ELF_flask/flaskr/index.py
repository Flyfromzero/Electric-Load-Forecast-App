from pandas import read_csv
import functools
from flask import (
    Blueprint, flash, g, redirect, render_template, request, session, url_for
)
from flaskr.utilities_pytorch import Dataset, ForecastLoad
from flaskr.db import get_db

bp = Blueprint('index', __name__, url_prefix='/index')

@bp.route('/predict', methods=('GET', 'POST'))
def predict():
    # press the button to generate results
    if request.method == 'POST':
        dataset = Dataset('flaskr/data/daily_dataset.csv')
        data = dataset.read_dataset()
        train, test = dataset.split_dataset(data)
        target = ['Usage']
        features = ['Generation','Net_Meter','Volt','Garage_E','Garage_W','Phase_A','Phase_B','Solar']
        _, target_scaler = dataset.normalize_dataset(train, target, features)
        test = dataset.weekly_split(test)
        X_test, y_test = dataset.to_supervised(test, n_input=7, n_output=7)
        X_test = dataset.to_tensor(X_test)
        forecast_load = ForecastLoad(model_path='flaskr/saved_cnnlstm_model.pth', data=X_test, target_scaler=target_scaler)
        results = forecast_load.forecast()
    return render_template("base.html",results=results)

