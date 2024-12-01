from flask import Flask,request,render_template
import numpy as np
from flask_cors import CORS
import pickle

model_crop = pickle.load(open('model_crop.pkl','rb'))
sc_crop = pickle.load(open('standscaler_crop.pkl','rb'))
mx_crop = pickle.load(open('minmaxscaler_crop.pkl','rb'))

model_fert = pickle.load(open('model_fert.pkl','rb'))
sc_fert = pickle.load(open('standscaler_fert.pkl','rb'))
mx_fert = pickle.load(open('minmaxscaler_fert.pkl','rb'))

app = Flask(__name__)

CORS(app)
@app.route('/', methods=['GET','POST'])
def index():
    return render_template("index.html")

@app.route("/predict",methods=['POST'])
def predict():
    N = request.form['Nitrogen']
    P = request.form['Phosporus']
    K = request.form['Potassium']
    temp = request.form['Temperature']
    humidity = request.form['Humidity']
    ph = request.form['pH']
    rainfall = request.form['Rainfall']
    EC = request.form['EC']

    crop_feature_list = [N, P, K, temp, humidity, ph, rainfall]
    crop_feature_array  = np.array(crop_feature_list).reshape(1, -1)

    mx_features_crop = mx_crop.transform(crop_feature_array)
    sc_mx_features_crop = sc_crop.transform(mx_features_crop)
    crop_prediction = model_crop.predict(sc_mx_features_crop)

    fert_feature_list = [N, P, K,ph, EC]
    fert_feature_array  = np.array(fert_feature_list).reshape(1, -1)

    mx_features_fert = mx_fert.transform(fert_feature_array)
    sc_mx_features_fert = sc_fert.transform(mx_features_fert)
    fert_prediction = model_fert.predict(sc_mx_features_fert)


    crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                 14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                 19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

    fert_dict = { 0:"Low", 1:"Moderate", 2:"High"}
    
    crop = crop_dict.get(crop_prediction[0], "Unknown Crop")
    fert = fert_dict.get(fert_prediction[0], "Unknown Fertility Level")

    crop_result = f"{crop}"
    fert_result = f"{fert}"
    
    return render_template('index.html',crop_result = crop_result, fert_result = fert_result)


if __name__ == "__main__":
    app.run(debug=True)