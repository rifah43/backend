
# Load the models
model_men = joblib.load('models/naive_bayes_men.pkl')
model_women = joblib.load('models/logistic_regression_women.pkl')

#  create a get req here
def predict_t2dm():
    data = request.json
    gender = data.get('Gender')
    features = pd.DataFrame([data])
    
    if gender == 'Male':
        prediction = model_men.predict(features[['Intensity', 'APQ11_Shimmer']])
    elif gender == 'Female':
        prediction = model_women.predict(features[['Pitch', 'Pitch_SD', 'RAP_Jitter']])
    else:
        return jsonify({'error': 'Invalid gender'}), 400
    
    result = 'T2DM' if prediction[0] == 1 else 'Non-T2DM'
    return jsonify({'prediction': result})