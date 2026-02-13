
from flask import Flask , render_template  ,request , url_for , redirect , session
import pickle
import numpy as np 
import pandas as pd 

app = Flask(__name__)

app.secret_key = "mlprojectsecret"

model = pickle.load(open('model.pkl' , 'rb'))
model_columns = pickle.load(open('columns.pkl', 'rb'))



@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None

    if request.method =='POST':
        if request.form['global_rank'] != "": 

            input_data = pd.DataFrame(columns=model_columns)
            input_data.loc[0] = 0
            
            input_data['global_rank'] = float(request.form['global_rank'])
            input_data['bounce_rate_pct'] = float(request.form['bounce_rate'])
            input_data['avg_session_duration_s'] = float(request.form['session_duration'])  
            input_data['stickiness_index'] = float(request.form['stickiness'])

            prediction = model.predict(input_data)[0]

            session['prediction'] = prediction

            return redirect(url_for('home'))
    prediction = session.pop('prediction', None)

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)

