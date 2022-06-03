# prediction function
#!/usr/bin/python -tt
import pickle
from socketserver import _RequestType
from urllib import request
from urllib.request import Request
from flask import Flask, render_template
from urllib import render_template

import numpy as np


def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1, 12)
    loaded_model = pickle.load(open("1.pys", "rb"))
    result = loaded_model.predict(to_predict)
    return result[0]


@pickle.APPEND.route('/result', methods=['POST'])
def result():
    if Request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(int, to_predict_list))
        result = ValuePredictor(to_predict_list)
        if int(result) == 1:
            prediction = 'Approve'
        else:
            prediction = 'Disapprove'
        return render_template("result.html", prediction=prediction)
