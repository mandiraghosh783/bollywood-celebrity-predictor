#!pip install mtcnn==0.1.0
# !pip install tensorflow==2.3.1
# !pip install keras==2.4.3
# !pip install keras-vggface==0.6
# !pip install keras_applications==1.0.8

import os
import pickle

actors = os.listdir('../bolly-celeb-predictor/data')

filenames = []

for actor in actors:
    for file in os.listdir(os.path.join('../bolly-celeb-predictor/data',actor)):
        filenames.append(os.path.join('../bolly-celeb-predictor/data',actor,file))

pickle.dump(filenames,open('filenames.pkl','wb'))
#import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import urllib
from urllib import request
from urllib.request import urlopen

from tensorflow.keras.preprocessing import image
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import numpy as np
import pickle
import ssl
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import certifi
#import ssl
#context = ssl._create_unverified_context()
#urllib.urlopen("https://no-valid-cert", context=context)
from tqdm import tqdm

filenames = pickle.load(open('filenames.pkl','rb'))

model = VGGFace(model='resnet50',include_top=False,input_shape=(224,224,3),pooling='avg')
def feature_extractor(img_path,model):
    img = image.load_img(img_path,target_size=(224,224))
    img_array = image.img_to_array(img)
    expanded_img = np.expand_dims(img_array,axis=0)
    preprocessed_img = preprocess_input(expanded_img)

    result = model.predict(preprocessed_img).flatten()

    return result
features = []

for file in tqdm(filenames):
    features.append(feature_extractor(file,model))

pickle.dump(features,open('embedding.pkl','wb'))


