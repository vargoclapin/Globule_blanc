from flask import Flask, request, render_template
import numpy as np

from programs.image import Image_Normalizer_VGG16
from programs.model_loader import ModelLoader

categories = {0:'Basophil',
              1:'Eosinophil',
              2:'Erythroblast',
              3:'Immature granulocyte',
              4:'Lymphocyte',
              5:'Monocyte',
              6:'Neutrophil',
              7:'Platelet'}

app = Flask(__name__, static_folder='static')

@app.route('/')
def login_page():
    return render_template('login.html')

@app.route('/upload', methods=['POST'])
def predict():
    
    if request.method == 'POST':
        
        # load the picture and load it into a normalised Pytorch tensor
        normalizer = Image_Normalizer_VGG16(request.files['myPicture'])
        tensor = normalizer.normalise()

        # handle model
        model = ModelLoader(request.form['model'])
        predictions = model.predict(tensor).detach().numpy()

        return render_template('result.html', result = categories[np.argmax(predictions)])

if __name__ == '__main__':
    app.run(debug=True)
