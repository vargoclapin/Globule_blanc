import torch

from programs.vgg16 import VGG16TransferLearning

class ModelLoader():

    def __init__(self, model_name: str):
        
        if model_name == 'tl':
            self._model = VGG16TransferLearning(num_classes=8)
            ckpt = torch.load('./model-tl/model_phase1.ckpt')
            self._model.load_state_dict(ckpt['state_dict'])
        if model_name == 'ft':
            self._model = VGG16TransferLearning(num_classes=8)
            ckpt = torch.load('./model-ft/model_phase2.ckpt')
            self._model.load_state_dict(ckpt['state_dict'])  
        
        # disable randomness, dropout, etc...
        self._model.eval()
        for param in self._model.model.features.parameters():
            param.requires_grad = False
        
    def predict(self, x):
        return self._model(x)