import numpy as np
import os
from keras.preprocessing import image
from werkzeug.utils import secure_filename
from keras.models import load_model
import tensorflow as tf
global graph
graph=tf.get_default_graph()
from flask import Flask,request,render_template
app=Flask(__name__)
model=load_model('project3.h5')
print('Model loaded. Check http://127.0.0.1:5000/')
@app.route('/',methods=['GET'])
def index():
	return render_template('base.html')
@app.route('/predict',methods=['GET','POST'])
def upload():
	if request.method=='POST':
		f=request.files['image']
	basepath=os.path.dirname(__file__)
	file_path=os.path.join(basepath,'uploads',secure_filename(f.filename))
	f.save(file_path)
	img=image.load_img(file_path,target_size=(64,64))
	x=image.img_to_array(img)
	x=np.expand_dims(x,axis=0)
	with graph.as_default:
		preds=model.predict_classes(x)
	index=['Igneous','Metamorphic','Sedimentary']

	text="prediction:"+index[preds[0]]
	return text
if __name__=='__main__':
	app.run(debug=True,threaded=False)
