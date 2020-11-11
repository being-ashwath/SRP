from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__)

model=pickle.load(open('model.pkl','rb'))
scale=pickle.load(open('model1.pkl','rb'))


@app.route('/')
def hello_world():
    return render_template('Home.html')
@app.route('/form')
def hello():
    return render_template('Form.html')
@app.route('/about')
def hello_w():
    return render_template('aboutD.html')
@app.route('/aboutus')
def hello_wo():
    return render_template('AboutUs.html')
@app.route('/contact')
def hello_wor():
    return render_template('Contact.html')
@app.route('/faq')
def hello_worl():
    return render_template('FAQ.html')

@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features=np.array([float(x) for x in request.form.values()])
    final=scale.transform(int_features.reshape(1,-1))
    print(final)
    prediction=model.predict(final)
    output=prediction

    if output==1:
        return render_template('Form.html',pred='Your Health is in Danger!\nYou are at risk of  diabetes.')
    else:
        return render_template('Form.html',pred='Your Health is safe!\n Keep up the good work!')

if __name__ == '__main__':
    app.run(port='5000',debug=False)
