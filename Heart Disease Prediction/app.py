from flask import Flask, render_template, request
import pickle 

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')



@app.route('/calculate', methods=['POST'])
def calculate():
    age=request.form['age']
    thalach= request.form['thalach']
    sex=request.form['sex']
    cp=request.form['cp']
    trestbps=request.form['trestbps']
    chol=request.form['chol']
    fps=request.form['fbs']
    restecg=request.form['restecg']
    exang=request.form['exang']
    oldpeak=float(request.form['oldpeak'])
    slope=request.form['slope']
    thal=request.form['thal']
    ca=request.form['ca']
    
    
    
    
    
    heart_model = pickle.load(open('heart2.pkl','rb'))

      

    
    
    pred = heart_model.predict([[age,thalach,sex,cp,trestbps,chol,fps,restecg,exang,oldpeak,slope,ca,thal]])
    
    return render_template('result.html', pred = pred)

if __name__ == '__main__':
    app.run(debug=True)