import os
from flask import Flask, request, jsonify 
import numpy as np 
from sklearn.linear_model import LinearRegression 

app2 = Flask(__name__) 

# Initialize a simple linear regression model 
model = LinearRegression() 
 
# Train the model with some dummy data 

X = np.array([[1], [2], [3], [4], [5]]) 
y = np.array([1, 2, 3, 4, 5]) 
model.fit(X, y) 

@app2.route('/predict', methods=['POST']) 
def predict(): 
    data = request.json 
    # Assuming the input is a list of values 
    values = np.array(data['values']).reshape(-1, 1) 
    predictions = model.predict(values) 
    return jsonify(predictions.tolist()) 

if __name__ == '__main2__': 
    os.environ['FLASK_ENV'] = 'development'
    app2.run(debug=True)