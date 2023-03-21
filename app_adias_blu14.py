import os
import json
import pickle
import joblib
import pandas as pd
from flask import Flask, jsonify, request
from peewee import (
    SqliteDatabase, PostgresqlDatabase, Model, IntegerField,
    FloatField, TextField, IntegrityError
)
from playhouse.shortcuts import model_to_dict

########################################
# Begin database stuff

DB = SqliteDatabase('predictions.db')


class Prediction(Model):
    observation_id = IntegerField(unique=True)
    observation = TextField()
    proba = FloatField()
    true_class = IntegerField(null=True)

    class Meta:
        database = DB


DB.create_tables([Prediction], safe=True)

# End database stuff
########################################



########################################
# Unpickle the previously-trained model


with open('columns.json') as fh:
    columns = json.load(fh)

pipeline = joblib.load('pipeline.pickle')

with open('dtypes.pickle', 'rb') as fh:
    dtypes = pickle.load(fh)


# End model un-pickling
########################################



########################################
# Input validation functions


# End input validation functions
########################################


########################################
# Begin webserver stuff

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():

    obs_dict = request.get_json()

    # test presence of 'id' in request
    if "observation_id" not in obs_dict:
        error = "Field `observation_id` missing from request: {}".format(obs_dict)
        return {"observation_id":None, "error": error}
    
    # test presence of 'data' in request   
    if "data" not in obs_dict:
        error = "Field `data` missing from request: {}".format(obs_dict)
        return {"observation_id":obs_dict['observation_id'], "error": error}


    _id = obs_dict['observation_id']
    observation = obs_dict['data']


    # test categorical features
    valid_category_map = {
                          "sex": ["Male", "Female"],
                          "race": ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"],
                         }
    
    for key, valid_categories in valid_category_map.items():
        if key in obs_dict['data'].keys():
            value = obs_dict['data'][key]
            if value not in valid_categories:
                error = "Invalid value provided for {}: {}. Allowed values are: {}".format(
                    key, value, ",".join(["'{}'".format(v) for v in valid_categories]))
                return {"observation_id":obs_dict['observation_id'], "error": error}  


    # test numerical features (age)
    age = obs_dict['data'].get("age")
    
    if not isinstance(age, int):
        error = "Invalid age: {}".format(age)
        return {"observation_id":obs_dict['observation_id'], "error": error}
    
    if age < 10 or age > 100:
        error = "Invalid age: {}".format(age)
        return {"observation_id":obs_dict['observation_id'], "error": error}
    

    # test numerical features (capital-gain and capital-loss)
    capital_gain = obs_dict['data'].get("capital-gain")
    capital_loss = obs_dict['data'].get("capital-loss")
    
    if capital_gain < 0:
        error = "Invalid capital-gain: {}".format(capital_gain)
        return {"observation_id":obs_dict['observation_id'], "error": error}
    
    if capital_loss < 0:
        error = "Invalid capital-loss: {}".format(capital_loss)
        return {"observation_id":obs_dict['observation_id'], "error": error}

    
    # test numerical features (hours-per-week)
    hours_per_week = obs_dict['data'].get("hours-per-week")
    
    if hours_per_week < 0 or hours_per_week > 168:
        error = "Invalid hours-per-week{}".format(hours_per_week)
        return {"observation_id":obs_dict['observation_id'], "error": error}


    # test valid columns
    valid_columns = {'age','workclass','education','marital-status',
                     'race','sex','capital-gain','capital-loss','hours-per-week'}
    
    keys = set(obs_dict['data'].keys())
    
    if len(valid_columns - keys) > 0:
        missing = [key for key in valid_columns if key not in keys]
        error = "Missing columns: {}".format(missing)
        return {"observation_id":obs_dict['observation_id'], "error":error}
    
    if len(keys - valid_columns) > 0: 
        extra = [key for key in keys if key not in valid_columns]
        error = "Unrecognized columns provided: {}".format(extra)
        return {"observation_id":obs_dict['observation_id'], "error": error}


    # make prediction
    obs = pd.DataFrame([observation], columns=columns).astype(dtypes)
    proba = pipeline.predict_proba(obs)[0, 1]
    prediction = pipeline.predict(obs)[0]
    response = {'prediction': bool(prediction), 'probability': proba}

    p = Prediction(
        observation_id=_id,
        proba=proba,
        observation=observation,
    )
    try:
        p.save()
    except IntegrityError:
        error_msg = "ERROR: Observation ID: '{}' already exists".format(_id)
        response["error"] = error_msg
        print(error_msg)
        DB.rollback()

    return jsonify(response)


@app.route('/update', methods=['POST'])
def update():
    obs = request.get_json()
    try:
        p = Prediction.get(Prediction.observation_id == obs['id'])
        p.true_class = obs['true_class']
        p.save()
        return jsonify(model_to_dict(p))
    except Prediction.DoesNotExist:
        error_msg = 'Observation ID: "{}" does not exist'.format(obs['id'])
        return jsonify({'error': error_msg})


@app.route('/list-db-contents')
def list_db_contents():
    return jsonify([
        model_to_dict(obs) for obs in Prediction.select()
    ])


# End webserver stuff
########################################
    
if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=5000)