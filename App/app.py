import os
from flask import Flask, render_template, request, jsonify
from PIL import Image
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as nn_func
from torchvision import transforms, models

from model.CNN import CNN

class PlantDiseaseApp:
    def __init__(self):
        self.app = Flask(__name__)
        self.model = self.loadModel()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.idxToClasses = self.loadClasses()
        self.diseaseInfo = pd.read_csv('data/disease_info.csv', encoding='cp1252')
        self.supplementInfo = pd.read_csv('data/supplement_info.csv', encoding='cp1252')
        self.registerRoutes()

    def loadModel(self):
        
        model = models.resnet50()
        model.fc = nn.Linear(model.fc.in_features, len(self.loadClasses()))
        model.load_state_dict(torch.load("model/transfer_learning.pth", map_location = torch.device('cpu')))

        # model = CNN(3, 38)
        # model.load_state_dict(torch.load("model/train_from_scratch.pth", map_location = torch.device('cpu')))
        
        model.eval()

        return model

    def loadClasses(self):
        return {
            0: 'Apple___Apple_scab', 1: 'Apple___Black_rot', 2: 'Apple___Cedar_apple_rust', 3: 'Apple___healthy',
            4: 'Background_without_leaves', 5: 'Blueberry___healthy', 6: 'Cherry___Powdery_mildew', 7: 'Cherry___healthy',
            8: 'Corn___Cercospora_leaf_spot Gray_leaf_spot', 9: 'Corn___Common_rust', 10: 'Corn___Northern_Leaf_Blight',
            11: 'Corn___healthy', 12: 'Grape___Black_rot', 13: 'Grape___Esca_(Black_Measles)', 14: 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
            15: 'Grape___healthy', 16: 'Orange___Haunglongbing_(Citrus_greening)', 17: 'Peach___Bacterial_spot',
            18: 'Peach___healthy', 19: 'Pepper,_bell___Bacterial_spot', 20: 'Pepper,_bell___healthy', 21: 'Potato___Early_blight',
            22: 'Potato___Late_blight', 23: 'Potato___healthy', 24: 'Raspberry___healthy', 25: 'Soybean___healthy',
            26: 'Squash___Powdery_mildew', 27: 'Strawberry___Leaf_scorch', 28: 'Strawberry___healthy',
            29: 'Tomato___Bacterial_spot', 30: 'Tomato___Early_blight', 31: 'Tomato___Late_blight', 32: 'Tomato___Leaf_Mold',
            33: 'Tomato___Septoria_leaf_spot', 34: 'Tomato___Spider_mites Two-spotted_spider_mite',
            35: 'Tomato___Target_Spot', 36: 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
            37: 'Tomato___Tomato_mosaic_virus', 38: 'Tomato___healthy'
        }

    def predict(self, filepath):
        try:
            img = Image.open(filepath)
            img = self.transform(img)

            with torch.no_grad():
                
                output = self.model(img.unsqueeze(0))
                predictedClass = torch.argmax(output)
    
                classesProbablities = nn_func.softmax(output, dim=1) * 100
                maxClassProbablity = round(max(classesProbablities[0]).item(), 2)

            return predictedClass.item(), maxClassProbablity

        except Exception as e:
            return jsonify({'error': str(e)})

    def registerRoutes(self):
        
        app = self.app

        @app.route('/')
        def homePage():
            return render_template('home.html')

        @app.route('/contact')
        def contact():
            return render_template('contact-us.html')

        @app.route('/index')
        def aiEnginePage():
            return render_template('index.html')

        @app.route('/submit', methods=['GET', 'POST'])
        def submit():
            if request.method == 'POST':
                image = request.files['image']
                filename = image.filename
                filePath = os.path.join('static/uploads', filename)
                image.save(filePath)
                pred, prob = self.predict(filePath)

                title = self.diseaseInfo['disease_name'][pred]
                description = self.diseaseInfo['description'][pred]
                prevent = self.diseaseInfo['Possible Steps'][pred]
                imageURL = self.diseaseInfo['image_url'][pred]
                supplementName = self.supplementInfo['supplement name'][pred]
                supplementImageURL = self.supplementInfo['supplement image'][pred]
                supplementBuyLink = self.supplementInfo['buy link'][pred]

                return render_template('submit.html', title=title, desc=description, prevent=prevent,
                                       image_url=imageURL, pred=pred, sname=supplementName,
                                       simage=supplementImageURL, buy_link=supplementBuyLink, probability=prob)

        @app.route('/market', methods=['GET', 'POST'])
        def market():
            return render_template('market.html',
                                   supplement_image=list(self.supplementInfo['supplement image']),
                                   supplement_name=list(self.supplementInfo['supplement name']),
                                   disease=list(self.diseaseInfo['disease_name']),
                                   buy=list(self.supplementInfo['buy link']),
                                   )


if __name__ == '__main__':
    appInstance = PlantDiseaseApp()
    appInstance.app.run(debug=True)
