import pandas as pd 
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense , BatchNormalization , LeakyReLU
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import ModelCheckpoint ,TensorBoard
import datetime 
from pickle import load
from sklearn.metrics import recall_score
from flask import Flask, jsonify, request
import flask

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Call index!'

@app.route('/index')
def index():
    return flask.render_template('index.html')

@app.route('/predict', methods=['POST'])
def final_fun_1() :
    ''' Takes pandas data frame as input and predict if the order is Back order or not'''
    to_predict = request.form.to_dict()
    # numbers as float
    to_predict = { key : ( float( value ) if ( all(map(str.isdigit, value)) and value != '' ) else value ) for (key, value) in to_predict.items() }
    columns = ['national_inv','lead_time','in_transit_qty','forecast_3_month','forecast_6_month','forecast_9_month','sales_1_month','sales_3_month','sales_6_month','sales_9_month',
               'min_bank','potential_issue','pieces_past_due','perf_6_month_avg','perf_12_month_avg','local_bo_qty','deck_risk','oe_constraint','ppap_risk','stop_auto_buy','rev_stop']
    X = [ to_predict.get(col) for col in columns ]
    X = pd.DataFrame([X], columns = columns)
    impute_leadtime = np.load('impute_leadtime.npy')
    X['lead_time'] = X['lead_time'].fillna(impute_leadtime)  #1st preprocessing
    #Replacing the boolean columns which has Yes with 1 and No with 0 
    for col in ['potential_issue','deck_risk', 'oe_constraint','ppap_risk', 'stop_auto_buy', 'rev_stop']:
        X[col] = X[col].map({'Yes': 1 , 'No' : 0})  
        
    if 'sku' in X.columns :
        X = X.drop(columns=['sku'])
    if 'went_on_backorder' in X.columns :
        X = X.drop(columns=['went_on_backorder'])  
    # The Custom nine fields     
    X['net_quantity'] = X.apply(lambda row: row.national_inv +  row.in_transit_qty , axis = 1)
    X['safe_quantity'] = X.apply(lambda row: row.net_quantity -  row.min_bank , axis = 1)
    X['safe_quantity_pos'] = np.where(X['safe_quantity'] >= 0, 1, 0)
    X['max_fore_cast_1_month'] = X.apply(lambda row: max( (row.forecast_9_month - row.forecast_6_month) /3, (row.forecast_6_month - row.forecast_3_month) /3 ) , axis = 1)
    X['min_fore_cast_1_month'] = X.apply(lambda row: min( (row.forecast_9_month - row.forecast_6_month) /3, (row.forecast_6_month - row.forecast_3_month) /3 ) , axis = 1)
    X['safe_max_diff'] = X.apply(lambda row: row.safe_quantity - row.max_fore_cast_1_month, axis = 1)
    X['safe_min_diff'] = X.apply(lambda row: row.safe_quantity - row.min_fore_cast_1_month, axis = 1)
    X['safe_max_diff_pos'] = np.where(X['safe_max_diff'] >= 0, 1, 0)
    X['safe_min_diff_pos'] = np.where(X['safe_min_diff'] >= 0, 1, 0) 
    
    scaled_X = X.copy(deep = True)
    scaler = load((open('scaler_new.pkl','rb')))
    col = ['national_inv', 'lead_time', 'in_transit_qty', 'forecast_3_month', 'forecast_6_month', 'forecast_9_month', 'sales_1_month', 'sales_3_month', 'sales_6_month', 'sales_9_month', 'min_bank', 'pieces_past_due', 'perf_6_month_avg', 'perf_12_month_avg', 'local_bo_qty', 'net_quantity', 'safe_quantity', 'max_fore_cast_1_month', 'min_fore_cast_1_month', 'safe_max_diff', 'safe_min_diff']
    features = scaled_X[col]
    features = scaler.transform(features.values)
    scaled_X[col] = features   
    #auto encoder features 
    tf.keras.backend.clear_session()
    # Building the Input Layer
    input_layer = Input(shape =(30,))

    # Building Encoder layer
    encoded = Dense(25)(input_layer)
    encoded = BatchNormalization()(encoded)
    encoded = LeakyReLU()(encoded)

    encoded = Dense(20)(encoded)
    encoded = BatchNormalization()(encoded)
    encoded = LeakyReLU()(encoded)

    encoded = Dense(15)(encoded)
    encoded = BatchNormalization()(encoded)
    encoded = LeakyReLU()(encoded)

    encoded = Dense(10)(encoded)
    encoded = BatchNormalization()(encoded)
    encoded = LeakyReLU()(encoded)

    # Building Decoder layer
    decoded = Dense(15)(encoded)
    decoded = BatchNormalization()(decoded)
    decoded = LeakyReLU()(decoded)

    decoded = Dense(20)(decoded)
    decoded = BatchNormalization()(decoded)
    decoded = LeakyReLU()(decoded)

    decoded = Dense(25)(decoded)
    decoded = BatchNormalization()(decoded)
    decoded = LeakyReLU()(decoded)

    # Building Output Layer
    output_layer = Dense(30, activation ='relu')(decoded)

    autoencoder = Model(input_layer, output_layer)
    autoencoder.load_weights('weights.hdf5') 

    encode = Sequential()
    encode.add(autoencoder.layers[0])
    encode.add(autoencoder.layers[1])
    encode.add(autoencoder.layers[2])
    encode.add(autoencoder.layers[3])
    encode.add(autoencoder.layers[4])
    encode.add(autoencoder.layers[5])
    encode.add(autoencoder.layers[6])
    encode.add(autoencoder.layers[7])
    encode.add(autoencoder.layers[8])
    encode.add(autoencoder.layers[9])
    encode.add(autoencoder.layers[10])
    encode.add(autoencoder.layers[11])
    encode.add(autoencoder.layers[12])    
    
    auto_encode = encode.predict(scaled_X)
    auto_encode_columns = ['auto_encode_'+str(i) for i in range(1,11)]
    auto_encode = pd.DataFrame(data = auto_encode , columns = auto_encode_columns )

    auto_encode = pd.concat([X.reset_index(drop=True), auto_encode.reset_index(drop=True)], axis=1) 
    
    final_model = load(open('final_rf_model.sav', 'rb'))
    pred = final_model.predict(auto_encode)

    if pred[0]:
        prediction = "Normal Order"
    else:
        prediction = "Back Order"

    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
