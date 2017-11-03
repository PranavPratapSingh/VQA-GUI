
import os, argparse
import cv2, spacy, numpy as np
from keras.models import model_from_json
from keras.optimizers import SGD
from sklearn.externals import joblib
from keras.applications import vgg16
import matplotlib.pyplot as plt
import sys

VQA_weights_file_name   = 'models/VQA/VQA_MODEL_WEIGHTS.hdf5'
label_encoder_file_name = 'models/VQA/FULL_labelencoder_trainval.pkl'
CNN_weights_file_name   = 'models/CNN/vgg16_weights (1).h5'
image_file_name = '/home/pranavpratapsingh/Documents/Codes/Python/PECFEST17/media/'+sys.argv[1]
#question = u'how are the people feeling?'
#question = u'how many people are there?'
#question = u'how many are there in the image?'
#question = u'what is the color of the bird?'
#question = u'is there a tree?'
question=unicode(sys.argv[2], "utf-8")

display_image = False

verbose = 0

def pop(model):
    if not model.outputs:
        raise Exception('Sequential model cannot be popped: model is empty.')
    else:
        model.layers.pop()
        if not model.layers:
            model.outputs = []
            model.inbound_nodes = []
            model.outbound_nodes = []
        else:
            model.layers[-1].outbound_nodes = []
            model.outputs = [model.layers[-1].output]
        model.built = False

    return model

def get_image_model(CNN_weights_file_name):
    model = vgg16.VGG16(include_top=True, weights='imagenet', input_shape=(224, 224, 3))
    model = pop(model)
    model = pop(model)
    return model

def get_image_features(image_file_name, CNN_weights_file_name):
    image_features = np.zeros((1, 4096))
    
    img = cv2.imread(image_file_name)
    # print image_file_name
    im = cv2.resize(img, (224, 224))

    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    # mean_pixel = [103.939, 116.779, 123.68]
    # mean_pixel = [123.68, 116.779, 103.939]

    im = im.astype(np.float32, copy=False)/255.0
    #for c in range(3):
    #    im[:, :, c] = im[:, :, c] - mean_pixel[c]

    if display_image:
        plt.imshow(im)#cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        plt.show()
        # exit(0)

    
    im = np.expand_dims(im, axis=0) 

    feature_model = get_image_model(CNN_weights_file_name)
    #im = vgg16.preprocess_input(im)
    # print get_image_model(CNN_weights_file_name).predict(im).shape, '-'*100
    image_features[0, :] = feature_model.predict(im)[0, :]
    return image_features

def get_VQA_model(VQA_weights_file_name):
    from models.VQA.VQA import VQA_MODEL
    vqa_model = VQA_MODEL()
    vqa_model.load_weights(VQA_weights_file_name)

    vqa_model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return vqa_model

def get_question_features(question):
    
    # word_embeddings = spacy.load('en_vectors_glove_md', vectors='en_glove_cc_300_1m_vectors')
    # word_embeddings = spacy.load('en', vectors='en_glove_cc_300_1m_vectors')
    word_embeddings = spacy.load('en', vectors='glove.6B.300d.txt')

    tokens = word_embeddings(question)
    question_tensor = np.zeros((1, 30, 300))
    for j in xrange(len(tokens)):
            question_tensor[0,j,:] = tokens[j].vector
    return question_tensor


def main():    
    if verbose : print("\n\n\nLoading image features ...")

    image_features = get_image_features(image_file_name, CNN_weights_file_name)

    if verbose : print("Loading question features ...")

    question_features = get_question_features(question)


    if verbose : print("Loading VQA Model ...")
    vqa_model = get_VQA_model(VQA_weights_file_name)


    if verbose : print("\n\n\nPredicting result ...") 
    y_output = vqa_model.predict([question_features, image_features])
    y_sort_index = np.argsort(y_output)

    labelencoder = joblib.load(label_encoder_file_name)
    for label in reversed(y_sort_index[0,-5:]):
        # vout = labelencoder.inverse_transform(label)
        print str(round(y_output[0,label]*100,2)).zfill(5), "% ", labelencoder.inverse_transform(label)

if __name__ == "__main__":
    main()
