#!/usr/local/bin/python3
from flask import render_template, request, jsonify
#import numpy as np
from lsmhun import APP
from imdb_ml_service import ImdbMlService

IMDB_ML_SERVICE = ImdbMlService()

@APP.route('/')
def movie_form():
    return render_template('movie_form.html')

@APP.route('/imdb_result', methods=['POST', 'GET'])
def result():
    global IMDB_ML_SERVICE
    if request.method == 'POST':
        content = request.get_json()
        #print(request.data)
        #print(content['comment'])
        #result_value = [[np.float32(0.92)]]
        result_value = IMDB_ML_SERVICE.evaluate_text(content['comment'])
        return jsonify(result_value[0][0].item())

if __name__ == '__main__':
    APP.run(debug=True, host='0.0.0.0')
