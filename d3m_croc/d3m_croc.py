import io
import os
import time
import re
import string
from json import dumps
from PIL import Image, ImageFilter

import requests
import spacy
import numpy as np
import pandas as pd
from keras.preprocessing import image
from keras.applications.inception_v3 \
    import InceptionV3, decode_predictions, preprocess_input

requests_session = requests.Session() if os.environ.get('USE_REQUESTS_SESSION') == "True" else requests

class Croc():

    def __init__(self, weights_path, spacy_path, isa_path, id_mapping_path):
        self.target_size = (299, 299)
        # self.model = InceptionV3(weights='imagenet')
        self.model = InceptionV3(weights=weights_path)
        # self.nlp = spacy.load('en_core_web_md')
        self.nlp = spacy_path
        self.n_top_preds = 10
        # self.isa_dict = isa_dict
        self.isa_dict = isa_path
        # self.id_mapping_dict = id_mapping_dict
        self.id_mapping_dict = id_mapping_path

    def load_image(self, img_path, prep_func=lambda x: x):
        ''' load image given path and convert to an array
        '''
        img = image.load_img(img_path, target_size=self.target_size)
        x = image.img_to_array(img)
        return prep_func(x)

    def load_image_from_web(self, image_url):
        ''' load an image from a provided hyperlink
        '''
        # get image
        response = requests_session.get(image_url)
        with Image.open(io.BytesIO(response.content)) as img:
            # fill transparency if needed
            if img.mode in ('RGBA', 'LA'):
                img = self.strip_alpha_channel(img)

            # convert to jpeg
            if img.format is not 'jpeg':
                img = img.convert('RGB')
            img.save('target_img.jpg')

    def validate_url(self, url):
        url_validator = re.compile(
            r'^(?:http|ftp)s?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|' #domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)

        return bool(url_validator.match(url))

    def cleanup_text(self, raw_chars):
        ''' use spacy to clean text and find tokens
        '''
        doc = self.nlp(raw_chars, disable=['parser', 'ner'])

        # grab raw text while removing any hanging newlines
        text = [t.text for t in doc if t.text.replace(' ', '').replace('\n', '')]

        tokens = [tok.lemma_.strip() for tok in doc
                  if self.nlp.vocab.has_vector(str(tok)) and # is_oov currently (v2.0.11) broken
                  tok.lemma_ != '-PRON-']

        tokens = [tok for tok in tokens
                  if tok not in self.nlp.Defaults.stop_words and
                  tok not in string.punctuation]

        return dict(tokens=list(set(tokens)), text=text)

    def classify_objects(self, image_array, decode_func):
        ''' Returns binary array with ones where the model predicts that
            the image contains an instance of one of the target classes
            (specified by wordnet id)
        '''
        predictions = self.model.predict(image_array)
        # decode the results into list of tuples (class, description, probability)
        predictions = decode_func(predictions, top=self.n_top_preds)
        return predictions

    def strip_alpha_channel(self, image, fill_color='#ffffff'):
        ''' Strip the alpha channel of an image and fill with fill color
        '''
        background = Image.new(image.mode[:-1], image.size, fill_color)
        background.paste(image, image.split()[-1])
        return background

    def char_detect(self, img_path):
        ''' Run tesseract ocr on an image supplied
            as an image path.
        '''
        with PyTessBaseAPI() as ocr_api:
            with Image.open(img_path) as image:
                # fill image alpha channel if it exists
                if image.mode in ('RGBA', 'LA'):
                    image = self.strip_alpha_channel(image)

                # will need a better preprocessing approach here
                # if we stay with tesseract:
                image = image.convert('L')
                # image = image.filter(ImageFilter.SHARPEN)
                # image = image.filter(ImageFilter.EDGE_ENHANCE)


                ocr_api.SetImage(image)
                raw_chars = ocr_api.GetUTF8Text()
                # char_confs = ocr_api.AllWordConfidences()

                text = self.cleanup_text(raw_chars)

                return text

    def climb_hierarchy(self, objects):

        def climb(self, object_, tree_):
            ''' function to climb the wordnet tree hierarchy and
                return the concepts along the branch.
            '''
            try:
                target_id = self.isa_dict[object_]
                target_label = self.id_mapping_dict[object_]
                tree_.append(dict(id=target_id, labels=target_label))
                climb(self, target_id, tree_)

            except Exception as e:
                e

            return tree_

        # trace branch to root for each object
        object_trees = list()
        for object_ in objects:
            tree_ = list()

            complete_tree = climb(self, object_, tree_)
            object_trees.extend(complete_tree)

        return object_trees

    def predict(self, input_path):
        ''' Produce predictions for objects and text
        '''
        image_path = input_path

        if self.validate_url(image_path):
            filename = 'target_img.jpg'
            self.load_image_from_web(image_path)
        else:
            filename = image_path

        print('preprocessing image')
        X = np.array(
            [self.load_image(
                filename, prep_func=preprocess_input)])

        print('making object predictions')
        object_predictions = self.classify_objects(X, decode_predictions)

        object_predictions = pd.DataFrame.from_records(
            object_predictions[0], columns=['id', 'label', 'confidence'])

        # object_trees = self.climb_hierarchy(objects=object_predictions['id'])

        # print('performing character recognition')
        # char_predictions = self.char_detect(filename)


        if filename == 'target_img.jpg':
            os.remove('target_img.jpg')

        return dumps(dict(
            objects=object_predictions.to_dict(),
            object_trees='',
            text='',
            tokens=''))


if __name__ == '__main__':
    inception_v3_path = '/home/croc_weights/inception_v3_weights_tf_dim_ordering_tf_kernels.h5' # location of inception v3 weights path
    isa_file_path = '/home/croc_weights/is_a.py' # location of file
    id_mapping_file_path = '/home/croc_weights/id_mapping_py.py' # location of file
    spacy_model = '/home/croc_weights/en_core_web_md-1.2.1.tar.gz' # location of spacy model 
    client = Croc(weights_path = inception_v3_path, spacy_path = spacy_model, isa_path = isa_file_path, id_mapping_path = id_mapping_file_path)
    image_path = 'http://farm4.static.flickr.com/3175/2737866473_7958dc8760.jpg'
    result = client.predict(input_path=image_path)
    print(result)
