
from flask import Flask , jsonify ,request,redirect,url_for
import numpy as np
import joblib
import tensorflow as tf
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config["CORS_SUPPORTS_CREDENTIALS"]=True
app.config["CORS_ALLOW_HEADERS"]=True
app.config["UPLOAD_FOLDER"] = "/home/usaibkhan/Desktop/Projects/FYP/RF-DOPD-Service/development/uploads"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

rf = joblib.load('rf.pkl')
symptomsDict=joblib.load('symptoms_dict.pkl')

def classifyImage(fileName):
    directory,filename = os.path.split( __file__ )
    classifier = tf.keras.models.load_model('model_4cat')
    img = tf.keras.preprocessing.image.load_img(directory+'/uploads/'+fileName, target_size=(150, 150))
    print(fileName)
    x = tf.keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x/255.0
    prediction = classifier.predict(x)
    return prediction
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/api/model/prediction",methods=['POST'])
def prediction():
    symptoms=request.json["symptoms"]
    inputVector=np.zeros(len(symptomsDict))
    for item in symptoms:
        inputVector[[symptomsDict[item]]]=1
    classes = rf.classes_
    predictions=rf.predict_proba([inputVector])
    n = 3
    top_n = np.argsort(predictions)[:,:-n-1:-1]
    results=[0,0,0]
    results[0]=classes[top_n[0][0]]
    results[1]=classes[top_n[0][1]]
    results[2]=classes[top_n[0][2]]
    print(results)
    return jsonify({"predictions": results})


@app.route("/api/model/uploadImage",methods=['POST'])
def uploadImage():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'photo' not in request.files:
            print('No file part')
            return jsonify({"msg": 'No file!'})
        file = request.files['photo']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            print('No selected file')
            return jsonify({"msg": 'please select file'})
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            predictions=classifyImage(filename)
            p=np.argmax(predictions)
            print(p,predictions)
            disease='Nothing Detected'
            if p==0:
                disease='Covid-19'
            if p==1:
                disease='Normal Chest'
            if p==2:
                disease='Pneumonia'
            if p==3:
                disease='Tuberculosis'
            return jsonify({"msg": 'file Uploaded Successfully','predictions':predictions.tolist(),'disease':disease}),200


if __name__=='__main__':
    app.run(debug=True,host='0.0.0.0')



