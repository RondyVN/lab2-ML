import pickle

import pandas as pd
from flask import Flask, render_template, request


app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def titanic_form():
    if request.method == 'GET':
        return render_template('form.html')
    else:
        res_dict = {
            'LotArea': request.form['LotArea'],
            'GarageArea': request.form['GarageArea'],
            'PoolArea': request.form['PoolArea'],
            'YearBuilt': request.form['YearBuilt'],
            'YearRemodAdd': request.form['YearRemodAdd'],
            'YrSold': request.form['YrSold']
        }
        print(res_dict)
        with open('result_of_train.pickle', 'rb') as pipline_file:
            predict = pickle.load(pipline_file).predict(pd.DataFrame(res_dict, index=[0]))[0]

        return render_template('result.html', predict=predict)

if __name__ == '__main__':
    app.run()
