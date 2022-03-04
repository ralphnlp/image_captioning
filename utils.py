from tqdm import tqdm
import pickle
import numpy as np
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.sequence import pad_sequences
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model


model = InceptionV3(weights='imagenet')
feature_extractor = Model(model.input, model.layers[-2].output)

with open('./models/pretrained_model.pkl', 'rb') as file:
    pretrain_model = pickle.load(file)

with open('./models/var.pkl', 'rb') as file:
    max_lengths, wordtoixs, ixtowords = pickle.load(file)


def preprocess(image_path):
    # Convert all the images to size 299x299 as expected by the inception v3 model
    img = image.load_img(image_path, target_size=(299, 299))
    # Convert PIL image to numpy array of 3-dimensions
    x = image.img_to_array(img)
    # Add one more dimension
    x = np.expand_dims(x, axis=0)
    # preprocess the images using preprocess_input() from inception module
    x = preprocess_input(x)
    return x

def encode(image):
    image = preprocess(image) # preprocess the image
    fea_vec = feature_extractor.predict(image) # Get the encoding vector for the image
    fea_vec = np.reshape(fea_vec, fea_vec.shape[1]) # reshape from (1, 2048) to (2048, )
    return fea_vec


def greedySearch(photo):
    in_text = 'startseq'
    for i in range(max_lengths):
        sequence = [wordtoixs[w] for w in in_text.split() if w in wordtoixs]
        sequence = pad_sequences([sequence], maxlen=max_lengths)
        yhat = pretrain_model.predict([photo,sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = ixtowords[yhat]
        in_text += ' ' + word
        if word == 'endseq':
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final


def img2caption(img_path):
    feature = encode(img_path)
    feature = feature.reshape((1,2048))
    caption = greedySearch(feature)
    return caption