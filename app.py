
from flask import Flask, render_template, request
import pickle
import numpy as np
#from flask_ngrok import run_with_ngrok
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
#run_with_ngrok(app)

model = pickle.load(open('model.pkl', 'rb'))

@app.route('/', methods=['GET'])
def home():
  return render_template('index.html')

@app.route('/', methods=['GET', "POST"])
def predict():
  input_values = [x for x in request.form.values()]
  inp_features = [input_values]
  print(inp_features) 
  original_features=[]
  for i in range(0,63):
    original_features.append(0)
  dict1={'mode_major':39,'mode_minor':40,'context_type_catalog':41,'context_type_charts':42,'context_type_editorial_playlist':43,'context_type_personalized_playlist':44,'context_type_radio':45,
        'context_type_user_collection':46,'hist_user_behavior_reason_start_appload':47,'hist_user_behavior_reason_start_backbtn':48,'hist_user_behavior_reason_start_clickrow':49,'hist_user_behavior_reason_start_endplay':50,
       'hist_user_behavior_reason_start_fwdbtn':51,
       'hist_user_behavior_reason_start_playbtn':52,
       'hist_user_behavior_reason_start_remote':53,
       'hist_user_behavior_reason_start_trackdone':54,
       'hist_user_behavior_reason_start_trackerror':55,
       'hist_user_behavior_reason_end_backbtn':56,
       'hist_user_behavior_reason_end_clickrow':57,
       'hist_user_behavior_reason_end_endplay':58,
       'hist_user_behavior_reason_end_fwdbtn':59,
       'hist_user_behavior_reason_end_logout':60,
       'hist_user_behavior_reason_end_remote':61,
       'hist_user_behavior_reason_end_trackdone':62}
  original_features[dict1[inp_features[0][39]]]=1
  original_features[dict1[inp_features[0][40]]]=1
  original_features[dict1[inp_features[0][41]]]=1
  #original_features[dict1[inp_features[0][42]]]=1
  for i in range(0,39):
    if inp_features[0][i]!='':
      original_features[i]=float(inp_features[0][i])
  original_features=[original_features]
  print(original_features)
  prediction = model.predict(original_features)
  if prediction==['Not Skipped']:
    return render_template('index.html', prediction_text='Prediction is Song is Not Skipped')
  else:
    return render_template('index.html', prediction_text='Prediction is Song is Skipped')

app.run(port=5000, debug=True)