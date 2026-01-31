import numpy as np 
import os 
import pickle
model_path = os.path.join(os.path.dirname(__file__),'model4.pkl')
def prediction(data):
    with open(model_path,'rb') as f:
        model = pickle.load(f)
    num_array = np.array(data).reshape(-1,1)
    pred = model.predict(num_array.T)
    return pred[0]



