from flask import Flask, render_template
import pickle
import numpy as np

model = pickle.load(open('best_model.pkl','rb'))

def create_app():
    app = Flask(__name__)

    @app.route('/')
    def homepage():
        return render_template('homepage.html')


    @app.route('/predict',methods=['POST'])
    def home():
        if request.method == 'POST':
            SpMax_L= request.form['a']  
            J_Dz_e= request.form['b']  
            nHM= request.form['c']  
            F01_NN = request.form['d']  
            F04_CN= request.form['e']  
            NssssC= request.form['f']  
            nCb= request.form['g']  
            Cpourcent= request.form['h']  
            nCp= request.form['i']  
            nO= request.form['j']  

            F03_CN= request.form['k']  
            SdssC= request.form['l']  
            HyWi_Bm= request.form['m']  
            LOC= request.form['n']  
            SM6_L= request.form['o']  
            F03_CO= request.form['p']  
            Me= request.form['q']  
            Mi= request.form['r']  
            nNN= request.form['s']  
            nArNO2= request.form['t']  

            nCRX3= request.form['u']  
            SpPosA_Bp= request.form['v']  
            nCIR= request.form['w']  
            B01C_Br= request.form['x']  
            B03_CCl= request.form['y']  
            N_073= request.form['z'] 
            SpMax_A= request.form['A1'] 

            Psi_i_1d= request.form['A2']  
            B04_CBr= request.form['A3']  
            SdO= request.form['A4']  
            TI2_L= request.form['A5']  

            nCrt= request.form['A6']  
            C_026= request.form['A7']  
            F02_CN= request.form['A8']  
            nHDon= request.form['A9']  
            SpMax_B_m= request.form['A10']  
            Psi_i_A= request.form['A11']  
            nN= request.form['A12']  
            SM6_Bm= request.form['A13']  
            nArCOOR= request.form['A14']  
            nX= request.form['A15']  



            element = pd.DataFrame([[SpMax_L,J_Dz_e,nHM,F01_NN,F04_CN, NssssC,nCb, Cpourcent, nCp, nO,F03_CN,SdssC, HyWi_Bm,LOC,SM6_L,F03_CO, Me,Mi,nNN, nArNO2,
            nCRX3,SpPosA_Bp,nCIR,B01C_Br,B03_CCl, N_073,SpMax_A,Psi_i_1d,B04_CBr,SdO,TI2_L,nCrt,C_026,F02_CN, nHDon,SpMax_B_m,Psi_i_A,nN, SM6_Bm,nArCOOR,nX]])
            print(element)
            response = model.predict(element)

            return render_template("response.html",data=response)   


    return app