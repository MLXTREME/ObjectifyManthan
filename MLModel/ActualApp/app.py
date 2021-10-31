"""
cd ActualApp
ObjEnv\Scripts\activate  
conda.bat deactivate   
python app.py
"""

from flask import Flask, render_template,request, url_for, send_from_directory, jsonify, send_file,redirect
import os
import zipfile
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField

from flask_wtf.file import FileField

class GreetUserForm(FlaskForm):
    photo = FileField(label=('Your photo'))
    submit = SubmitField(label=('Submit'))



# "templates" this is for plain html files or "Great_Templates" this is for complex css + imgs +js +html+sass
TEMPLATES = "templates"

app = Flask(__name__, static_folder="assets", template_folder=TEMPLATES)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5 MB Standard File Size

app.config['SECRET_KEY']='LongAndRandomSecretKey'
# ROOT_DIR = os.getcwd()
# ROOT_DIR = app.instance_path
ROOT_DIR = app.root_path
# Reloading
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

@app.route('/',methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')
@app.route('/test', methods=('GET', 'POST'))
def test():
    form = GreetUserForm()
    if form.validate_on_submit():
          return f'''<h1> Welcome </h1>'''
    
    return render_template('test.html', form=form)
 
        
 
    
         
        
    

@app.route('/results')
def upload_excel_file():
    return render_template('results.html')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'assets', 'favicons'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')


if __name__ == "__main__":
    app.run(debug=True) # debug=True