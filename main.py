from flask import Flask
from flask import Flask, render_template, Response, redirect, request, session, abort, url_for
import os
import base64
from PIL import Image
from datetime import datetime
from datetime import date
import datetime
import random
from random import seed
from random import randint
import re
import cv2
import PIL.Image
from PIL import Image
from flask import send_file
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import threading
import time
import shutil
import hashlib
import urllib.request
import urllib.parse
from urllib.request import urlopen
import webbrowser
import json
import mysql.connector

import gensim
from gensim.parsing.preprocessing import remove_stopwords, STOPWORDS
from gensim.parsing.porter import PorterStemmer
#import spacy
#nlp = spacy.load('en')

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  passwd="",
  charset="utf8",
  database="schemebot"
)


app = Flask(__name__)
##session key
app.secret_key = 'abcdef'
UPLOAD_FOLDER = 'static/upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#####

@app.route('/',methods=['POST','GET'])
def index():
    msg=""
    mycursor = mydb.cursor()

    ff=open("static/det.txt","w")
    ff.write("1")
    ff.close()

                
    if request.method == 'POST':
        
        username1 = request.form['uname']
        password1 = request.form['pass']
        mycursor = mydb.cursor()
        mycursor.execute("SELECT count(*) FROM cc_register where uname=%s && pass=%s",(username1,password1))
        myresult = mycursor.fetchone()[0]
        if myresult>0:
            session['username'] = username1
            #result=" Your Logged in sucessfully**"
            return redirect(url_for('bot')) 
        else:
            msg="You are logged in fail!!!"

    return render_template('index.html',msg=msg)

@app.route('/login',methods=['POST','GET'])
def login():
    cnt=0
    act=""
    msg=""
    if request.method == 'POST':
        
        username1 = request.form['uname']
        password1 = request.form['pass']
        mycursor = mydb.cursor()
        mycursor.execute("SELECT count(*) FROM cc_admin where username=%s && password=%s",(username1,password1))
        myresult = mycursor.fetchone()[0]
        if myresult>0:
            session['username'] = username1
            #result=" Your Logged in sucessfully**"
            return redirect(url_for('admin')) 
        else:
            msg="You are logged in fail!!!"
        

    return render_template('login.html',msg=msg,act=act)

@app.route('/login_user',methods=['POST','GET'])
def login_user():
    cnt=0
    act=""
    msg=""
    if request.method == 'POST':
        
        username1 = request.form['uname']
        password1 = request.form['pass']
        mycursor = mydb.cursor()
        mycursor.execute("SELECT count(*) FROM cc_register where uname=%s && pass=%s",(username1,password1))
        myresult = mycursor.fetchone()[0]
        if myresult>0:
            session['username'] = username1
            #result=" Your Logged in sucessfully**"
            return redirect(url_for('bot')) 
        else:
            msg="You are logged in fail!!!"
        

    return render_template('login_user.html',msg=msg,act=act)

@app.route('/admin', methods=['GET', 'POST'])
def admin():
    msg=""
    mycursor = mydb.cursor()
    if request.method=='POST':
        file = request.files['file']

        fn="datafile.csv"
        file.save(os.path.join("static/upload", fn))

        filename = 'static/upload/datafile.csv'
        data1 = pd.read_csv(filename, header=0,encoding='cp1252')
        data2 = list(data1.values.flatten())
        for ss in data1.values:

            des=""
            if pd.isnull(ss[2]):
                des=""
            else:
                des=ss[2]
                
            eligi=""
            if pd.isnull(ss[3]):
                eligi=""
            else:
                eligi=ss[3]
                
            '''mycursor.execute("SELECT max(id)+1 FROM cc_data")
            maxid = mycursor.fetchone()[0]
            if maxid is None:
                maxid=1

            

            sql = "INSERT INTO cc_data(id,scheme,department,description,eligibility) VALUES (%s,%s,%s,%s,%s)"
            val = (maxid,ss[0],ss[1],des,eligi)
            mycursor.execute(sql, val)
            mydb.commit()'''
        
        msg="success"


    return render_template('admin.html',msg=msg)

@app.route('/process', methods=['GET', 'POST'])
def process():
    msg=""
    cnt=0
    

    filename = 'static/upload/datafile.csv'
    data1 = pd.read_csv(filename, header=0,encoding='cp1252')
    data2 = list(data1.values.flatten())

    
    data=[]
    i=0
    sd=len(data1)
    rows=len(data1.values)
    
    #print(str(sd)+" "+str(rows))
    for ss in data1.values:
        cnt=len(ss)
        data.append(ss)
    cols=cnt

    
    return render_template('process.html',data=data, msg=msg, rows=rows, cols=cols)

@app.route('/process2', methods=['GET', 'POST'])
def process2():
    msg=""
    act=request.args.get("act")
    
    return render_template('process2.html',msg=msg, act=act)

@app.route('/add_query', methods=['GET', 'POST'])
def add_query():
    msg=""
    sid=""
    mycursor = mydb.cursor()

    cnt=0
    
    data=[]
    

    mycursor.execute("SELECT * FROM cc_data")
    data = mycursor.fetchall()
        
    
    if request.method=='POST':
        sid=request.form['sid']
        
        msg="success"

    return render_template('add_query.html',msg=msg,sid=sid,data=data)


@app.route('/add_query1', methods=['GET', 'POST'])
def add_query1():
    msg=""
    act=request.args.get("act")
    sid=request.args.get("sid")
    mycursor = mydb.cursor()
    
    cnt=0
    #filename = 'static/upload/datafile.csv'
    #data1 = pd.read_csv(filename, header=0,encoding='cp1252')
    #data2 = list(data1.values.flatten())

    
    data=[]
    #i=0
    #for ss in data1.values:
    #    cnt=len(ss)
    #    data.append(ss)
    #cols=cnt

    mycursor.execute("SELECT * FROM cc_data where id=%s",(sid,))
    data = mycursor.fetchone()

    mycursor.execute("SELECT * FROM cc_contact where scheme_id=%s",(sid,))
    data2 = mycursor.fetchall()
        
    
    if request.method=='POST':
        
        user_query=request.form['user_query']
        district=request.form['district']
        name=request.form['name']
        mobile=request.form['mobile']
        designation=request.form['designation']
        address=request.form['address']
        url_link=request.form['url_link']
        
        
        mycursor.execute("update cc_data set user_query=%s,district=%s,name=%s,mobile=%s,designation=%s,address=%s,url_link=%s where id=%s",(user_query,district,name,mobile,designation,address,url_link,sid))
        mydb.commit()

        mycursor.execute("SELECT count(*) FROM cc_contact where id=%s && name=%s && mobile=%s",(sid,name,mobile))
        d1 = mycursor.fetchone()[0]
        if d1==0:
            mycursor.execute("SELECT max(id)+1 FROM cc_contact")
            maxid = mycursor.fetchone()[0]
            if maxid is None:
                maxid=1

            

            sql = "INSERT INTO cc_contact(id,scheme_id,district,name,mobile,designation,address,url_link) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)"
            val = (maxid,sid,district,name,mobile,designation,address,url_link)
            mycursor.execute(sql, val)
            mydb.commit()
        else:
            mycursor.execute("update cc_contact set district=%s,name=%s,mobile=%s,designation=%s,address=%s,url_link=%s where sid=%s && name=%s",(district,name,mobile,designation,address,url_link,sid,name))
            mydb.commit()
        msg="success"

    if act=="del":
        did=request.args.get("did")
        mycursor.execute("delete from cc_contact where id=%s",(did,))
        mydb.commit()
        return redirect(url_for('add_query1',sid=sid)) 

    return render_template('add_query1.html',msg=msg,sid=sid,act=act,data=data,data2=data2)

@app.route('/admin2', methods=['GET', 'POST'])
def admin2():
    msg=""
    mycursor = mydb.cursor()
    if request.method=='POST':
        input1=request.form['input']
        output=request.form['output']
        link=request.form['link']

        if link is None or link=="":
            url=""
        else:
            url=' <a href='+link+' target="_blank">Click Here</a>'

        output+=url
        
        mycursor.execute("SELECT max(id)+1 FROM cc_data")
        maxid = mycursor.fetchone()[0]
        if maxid is None:
            maxid=1

        sql = "INSERT INTO cc_data(id,input,output) VALUES (%s,%s,%s)"
        val = (maxid,input1,output)
        mycursor.execute(sql, val)
        mydb.commit()

        
        print(mycursor.rowcount, "Added Success")
        
        return redirect(url_for('view_data',msg='success'))

    return render_template('admin2.html',msg=msg)

@app.route('/view_user', methods=['GET', 'POST'])
def view_user():
    value=[]
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM cc_register")
    data = mycursor.fetchall()

    
    return render_template('view_user.html', data=data)

@app.route('/page', methods=['GET', 'POST'])
def page():
    fn=request.args.get("fn")
   
    
    return render_template('page.html',fn=fn)



@app.route('/register',methods=['POST','GET'])
def register():
    msg=""
    act=""
    mycursor = mydb.cursor()
    name=""
    mobile=""
    mess=""
    uid=""
    if request.method=='POST':
        
        uname=request.form['uname']
        name=request.form['name']     
        mobile=request.form['mobile']
        email=request.form['email']
        location=request.form['location']
        pass1=request.form['pass']

        
        now = datetime.datetime.now()
        rdate=now.strftime("%d-%m-%Y")
        mycursor = mydb.cursor()

        mycursor.execute("SELECT count(*) FROM cc_register where uname=%s",(uname, ))
        cnt = mycursor.fetchone()[0]
        if cnt==0:
            mycursor.execute("SELECT max(id)+1 FROM cc_register")
            maxid = mycursor.fetchone()[0]
            if maxid is None:
                maxid=1
            
            uid=str(maxid)
            sql = "INSERT INTO cc_register(id, name, mobile, email, location,uname, pass,otp,status) VALUES (%s, %s, %s, %s, %s, %s, %s,%s,%s)"
            val = (maxid, name, mobile, email, location, uname, pass1,'','0')
            msg="success"
            mycursor.execute(sql, val)
            mydb.commit()            
            print(mycursor.rowcount, "record inserted.")
           
        else:
            msg="fail"
            
    return render_template('register.html',msg=msg,mobile=mobile,name=name,mess=mess,uid=uid)

#RandomForest
def RandomForestClassifier():
        
        estimator_params=tuple()
        bootstrap=False
        oob_score=False
        n_jobs=None
        random_state=None
        verbose=0
        warm_start=False
        class_weight=None
        max_samples=None
    
        super().__init__(
            estimator=estimator,
            n_estimators=n_estimators,
            estimator_params=estimator_params,
        )

        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start
        self.class_weight = class_weight
        self.max_samples = max_samples
        

def fit():
       
    if issparse(y):
        raise ValueError("sparse multilabel-indicator for y is not supported.")

    X, y = self._validate_data(
        X,
        y,
        multi_output=True,
        accept_sparse="csc",
        dtype=DTYPE,
        force_all_finite=False,
    )

    estimator = type(self.estimator)(criterion=self.criterion)
    missing_values_in_feature_mask = (
        estimator._compute_missing_values_in_feature_mask(
            X, estimator_name=self.__class__.__name__
        )
    )

    if sample_weight is not None:
        sample_weight = _check_sample_weight(sample_weight, X)

    y = np.atleast_1d(y)

    self._n_samples, self.n_outputs_ = y.shape

    y, expanded_class_weight = self._validate_y_class_weight(y)

    if getattr(y, "dtype", None) != DOUBLE or not y.flags.contiguous:
        y = np.ascontiguousarray(y, dtype=DOUBLE)

    if expanded_class_weight is not None:
        if sample_weight is not None:
            sample_weight = sample_weight * expanded_class_weight
        else:
            sample_weight = expanded_class_weight

    if not self.bootstrap and self.max_samples is not None:
        raise ValueError()
    elif self.bootstrap:
        n_samples_bootstrap = _get_n_samples_bootstrap(
            n_samples=X.shape[0], max_samples=self.max_samples
        )
    else:
        n_samples_bootstrap = None

    self._n_samples_bootstrap = n_samples_bootstrap

    self._validate_estimator()

    if not self.bootstrap and self.oob_score:
        raise ValueError("Out of bag estimation only available if bootstrap=True")

    random_state = check_random_state(self.random_state)

    if not self.warm_start or not hasattr(self, "estimators_"):
        # Free allocated memory, if any
        self.estimators_ = []

    n_more_estimators = self.n_estimators - len(self.estimators_)


def model():
    # Importing the dataset
    dataset = pd.read_csv('static/upload/datafile.csv')
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 1].values

    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

    # Train the model

    # random forest model (or any other preferred algorithm)
    from sklearn.ensemble import RandomForestRegressor
    regressor = RandomForestRegressor(n_estimators=20, random_state=0)

    regressor.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = regressor.predict(X_test)

    # Saving model using pickle
    pickle.dump(regressor, open('model.pkl','wb'))

    # Loading model to compare the results
    model = pickle.load( open('model.pkl','rb'))
    print(model.predict([[1.8]]))

            
@app.route('/bot', methods=['GET', 'POST'])
def bot():
    msg=""
    output=""
    uname=""
    mm=""
    s=""
    xn=0
    qry_st=""
    if 'username' in session:
        uname = session['username']
    mydb = mysql.connector.connect(
      host="localhost",
      user="root",
      passwd="",
      charset="utf8",
      database="schemebot"
    )

    
    
    cnt=0
   
    mycursor = mydb.cursor()

    mycursor.execute("SELECT * FROM cc_register where uname=%s",(uname, ))
    value = mycursor.fetchone()
    
    mycursor.execute("SELECT * FROM cc_data order by rand() limit 0,10")
    data=mycursor.fetchall()
            
    if request.method=='POST':
        msg_input=request.form['msg_input']
        
        text=msg_input

        ff=open("static/det.txt","r")
        qry_st=ff.read()
        ff.close()
        ##
        #NLP
        #nlp=STOPWORDS
        #def remove_stopwords(text):
        #    clean_text=' '.join([word for word in text.split() if word not in nlp])
        #    return clean_text
        ##
        #txt=remove_stopwords(msg_input)
        ##
        stemmer = PorterStemmer()
    
        from wordcloud import STOPWORDS
        STOPWORDS.update(['rt', 'mkr', 'didn', 'bc', 'n', 'm', 
                          'im', 'll', 'y', 've', 'u', 'ur', 'don', 
                          'p', 't', 's', 'aren', 'kp', 'o', 'kat', 
                          'de', 're', 'amp', 'will'])

        def lower(text):
            return text.lower()

        def remove_specChar(text):
            return re.sub("#[A-Za-z0-9_]+", ' ', text)

        def remove_link(text):
            return re.sub('@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+', ' ', text)

        def remove_stopwords(text):
            return " ".join([word for word in 
                             str(text).split() if word not in STOPWORDS])

        def stemming(text):
            return " ".join([stemmer.stem(word) for word in text.split()])

        #def lemmatizer_words(text):
        #    return " ".join([lematizer.lemmatize(word) for word in text.split()])

        def cleanTxt(text):
            text = lower(text)
            text = remove_specChar(text)
            text = remove_link(text)
            text = remove_stopwords(text)
            text = stemming(text)
            
            return text

        

        #show the clean text
        #dat=df.head()
        #data=[]
        #for ss in dat.values:
        #    data.append(ss)
        #msg_input=data
        mm=""
        mm1=""
        ######################
        if msg_input=="":
            s=1
            output="How can i help you?"
            return json.dumps(output)
        else:
            if qry_st=="1":
                clean_msg=cleanTxt(msg_input)
                print(clean_msg)
                cleaned='%'+clean_msg+'%'
                
                mycursor.execute("SELECT count(*) FROM cc_data where user_query like %s",(cleaned,))
                cnt1=mycursor.fetchone()[0]
                if cnt1>0:
                    mm='%'+clean_msg+'%'
                else:
                    mm='%'+msg_input+'%'

                ###
                mycursor.execute("SELECT count(*) FROM cc_data where scheme like %s",(cleaned,))
                cnt12=mycursor.fetchone()[0]
                if cnt12>0:
                    mm1='%'+clean_msg+'%'
                else:
                    mm1='%'+msg_input+'%'
                ###
                
                mycursor.execute("SELECT count(*) FROM cc_data where user_query like %s",(mm,))
                cnt=mycursor.fetchone()[0]



                
                if cnt>0:
                    dd3=""
                    sid=0
                    mycursor.execute("SELECT * FROM cc_data where user_query like %s limit 0,1",(mm,))
                    dd=mycursor.fetchall()
                    for dd1 in dd:
                        sid=dd1[0]
                        
                        dd3+="<br>"+dd1[1]+"<br><br>Department:<br>"+dd1[2]

                        if dd1[3]=="":
                            s=1
                        else:
                            
                            dd3+="<br><br>Relevant Component:<br>"+dd1[3]


                        if dd1[4]=="":
                            s=1
                        else:
                            dd3+="<br><br>Eligibility:<br>"+dd1[4]

                    dff=[]
                    dff2=""
                    
                    mycursor.execute("SELECT count(*) FROM cc_contact where scheme_id=%s",(sid,))
                    cnt4=mycursor.fetchone()[0]
                    if cnt4>0:
                        mycursor.execute("SELECT * FROM cc_contact where scheme_id=%s",(sid,))
                        dd4=mycursor.fetchall()
                        for dd41 in dd4:
                            dff.append(dd41[2])
                        dff2=",".join(dff)
                        dd3+="<br><br>Which location contacts you want this scheme?<br>"
                        dd3+="("+dff2+")"
                        ff=open("static/det.txt","w")
                        ff.write("2")
                        ff.close()

                        ff=open("static/scheme.txt","w")
                        ff.write(str(sid))
                        ff.close()

                    
                    output=dd3
                    
                

                else:
                    ####mm1
                    print("aa")
                    print(mm1)
                    dd3=""
                    sid=0
                    mycursor.execute("SELECT count(*) FROM cc_data where scheme like %s",(mm1,))
                    cnt11=mycursor.fetchone()[0]
                    if cnt11>0:
                                        
                        mycursor.execute("SELECT * FROM cc_data where scheme like %s limit 0,1",(mm1,))
                        ddx=mycursor.fetchall()
                        for dd1 in ddx:
                            sid=dd1[0]
                            print(dd1[1])
                            dd3+="<br>"+dd1[1]+"<br><br>Department:<br>"+dd1[2]

                            if dd1[3]=="":
                                s=1
                            else:
                                
                                dd3+="<br><br>Relevant Component:<br>"+dd1[3]


                            if dd1[4]=="":
                                s=1
                            else:
                                dd3+="<br><br>Eligibility:<br>"+dd1[4]

                        dff=[]
                        dff2=""
                        
                        mycursor.execute("SELECT count(*) FROM cc_contact where scheme_id=%s",(sid,))
                        cnt4=mycursor.fetchone()[0]
                        if cnt4>0:
                            mycursor.execute("SELECT * FROM cc_contact where scheme_id=%s",(sid,))
                            dd4=mycursor.fetchall()
                            for dd41 in dd4:
                                dff.append(dd41[2])
                            dff2=",".join(dff)
                            dd3+="<br><br>Which location contacts you want this scheme?<br>"
                            dd3+="("+dff2+")"
                            ff=open("static/det.txt","w")
                            ff.write("2")
                            ff.close()

                            ff=open("static/scheme.txt","w")
                            ff.write(str(sid))
                            ff.close()

                                        
                        output=dd3
                    
                    else:                    
                        if msg_input=="":
                            output="How can i help you?"
                        else:
                            output="Sorry, No Results Found!"

                return json.dumps(output)
                ####################

            

            elif qry_st=="2":
                clean_msg=cleanTxt(msg_input)
                print(clean_msg)
                cleaned='%'+clean_msg+'%'
                
                mycursor.execute("SELECT count(*) FROM cc_contact where district like %s",(cleaned,))
                cnt1=mycursor.fetchone()[0]
                if cnt1>0:
                    mm='%'+clean_msg+'%'
                else:
                    mm='%'+msg_input+'%'
                
                
                mycursor.execute("SELECT count(*) FROM cc_contact where district like %s",(mm,))
                cnt=mycursor.fetchone()[0]



                
                if cnt>0:
                    dd3=""
                    ff=open("static/scheme.txt","r")
                    sidd=ff.read()
                    ff.close()
                    mycursor.execute("SELECT * FROM cc_contact where district like %s && scheme_id=%s limit 0,1",(mm,sidd))
                    dd=mycursor.fetchall()
                    for dd1 in dd:
                        dd3+="<br>District: "+dd1[2]+"<br>Name: "+dd1[3]+"<br>Designation: "+dd1[5]
                        dd3+="<br>Address: "+dd1[6]+"<br>Mobile No.: "+str(dd1[4])
                        if dd1[7]=="":
                            s=1
                        else:
                            dd3+="<br><a href='"+dd1[7]+"' target='_blank'>Click Here</a>"
                       
                        
                    output=dd3
                    ff=open("static/det.txt","w")
                    ff.write("1")
                    ff.close()
                

                else:
                    if msg_input=="":
                        output="How can i help you?"
                    else:
                        output="Sorry, No Results Found!"


            
                        
                return json.dumps(output)
                ##################################


    return render_template('bot.html', msg=msg,output=output,uname=uname,data=data,value=value)   

    
@app.route('/sign')
def sign():
    return render_template('sign.html')

@app.route('/signUpUser', methods=['POST'])
def signUpUser():
    user =  request.form['username'];
    password = request.form['password'];

    print(password)
    return json.dumps({'status':'OK','user':user,'pass':password});


@app.route('/view_data', methods=['GET', 'POST'])
def view_data():
    data=[]
    msg=request.args.get("msg")
    act=request.args.get("act")
    url=""
    mycursor = mydb.cursor()

    mycursor.execute("SELECT * FROM cc_data")
    data1 = mycursor.fetchall()

    for dat in data1:
        dt=[]
        txt=dat[2]
        t=txt.replace("\t\r\n","<br>")
        #if "\t\r\n" in dat[2]:

        dt.append(dat[0])
        dt.append(dat[1])
        dt.append(t)
        data.append(dt)
            
            
    

    if request.method=='POST':
        input1=request.form['input']
        output=request.form['output']
        link=request.form['link']

        if link is None:
            url=""
        else:
            url=' <a href='+link+' target="_blank">Click Here</a>'

        output+=url
        
        mycursor.execute("SELECT max(id)+1 FROM cc_data")
        maxid = mycursor.fetchone()[0]
        if maxid is None:
            maxid=1

        sql = "INSERT INTO cc_data(id,input,output) VALUES (%s,%s,%s)"
        val = (maxid,input1,output)
        mycursor.execute(sql, val)
        mydb.commit()

        
        print(mycursor.rowcount, "Added Success")
        
        return redirect(url_for('view_data',msg='success'))
        #if cursor.rowcount==1:
        #    return redirect(url_for('index',act='1'))

    if act=="del":
        did=request.args.get("did")
        mycursor.execute("delete from cc_data where id=%s",(did,))
        mydb.commit()
        return redirect(url_for('view_data'))
    
    
    return render_template('view_data.html',msg=msg,act=act,data=data)

@app.route('/down', methods=['GET', 'POST'])
def down():
    fn = request.args.get('fname')
    path="static/upload/"+fn
    return send_file(path, as_attachment=True)

@app.route('/logout')
def logout():
    # remove the username from the session if it is there
    session.pop('username', None)
    return redirect(url_for('index'))


if __name__ == "__main__":
    app.secret_key = os.urandom(12)
    app.run(debug=True,host='0.0.0.0', port=5000)
