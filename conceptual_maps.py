import spacy

# import pandas as pd
# import numpy as np 

# import matplotlib.pyplot as plt

# Library to implement BERT
from summarizer import Summarizer

from langdetect import detect, detect_langs

import nltk
import nltk.data

# Deepl works especially fine with Europeans languages, but its free version is limited
# from deep_translator import DeeplTranslator
from deep_translator import GoogleTranslator


import click

# CLI parameters
@click.command()
@click.option('--data', '-d', default='data/input.txt', required=False, show_default=True,
              help=u'Input text file route')


def run_conceptual_maps(data):
    f = open(data)

    sections = f.readlines()

    # print(input)

    nltk.download('punkt')
    
    for input in sections:

        detected_lang = detect(input)
      
        if detected_lang == 'es':
            tokenizer = nltk.data.load('tokenizers/punkt/spanish.pickle')
        elif detected_lang == 'it':
            tokenizer = nltk.data.load('tokenizers/punkt/italian.pickle')

        # We won't need to separate English text yet 
        # elif detected_lang == 'en':
        #   tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

        if detected_lang != 'en':
            sentences = tokenizer.tokenize(input)
        # print ('\n-----\n'.join(tokenizer.tokenize(input)))

        # print(sentences)

        translations = []

        # If the detected language is English we won't need to translate it
        if detected_lang != 'en':
            # Splitting the text into sentences we ensure not reaching the character limit of Google Translator
            print('Translating text to English...')
            for sent in sentences:
                translations.append(GoogleTranslator(source='auto', target='en').translate(sent)) 
            print(translations)


        # If the detected language is English we have it already joined from input
        if detected_lang != 'en':
            full_translation = " ".join(w for word in translations for w in word.split())
        else:
            full_translation = input

        print(full_translation)

        bert_summarizer = Summarizer()

        # Used for testing the annalysis performance
        full_translation_test = 'Artificial neural networks (ANNs), usually simply called neural networks (NNs), are computing systems inspired by the biological neural networks that constitute animal brains. An ANN is based on a collection of connected units or nodes called artificial neurons, which loosely model the neurons in a biological brain. Each connection, like the synapses in a biological brain, can transmit a signal to other neurons. An artificial neuron receives a signal then processes it and can signal neurons connected to it. The "signal" at a connection is a real number, and the output of each neuron is computed by some non-linear function of the sum of its inputs. The connections are called edges. Neurons and edges typically have a weight that adjusts as learning proceeds. The weight increases or decreases the strength of the signal at a connection. Neurons may have a threshold such that a signal is sent only if the aggregate signal crosses that threshold. Typically, neurons are aggregated into layers. Different layers may perform different transformations on their inputs. Signals travel from the first layer (the input layer), to the last layer (the output layer), possibly after traversing the layers multiple times.'

        resolved_doc = full_translation
        print(len(resolved_doc.split()))
        summarize = bert_summarizer(resolved_doc, ratio=0.2)  # Specified with ratio
        print(summarize, len(summarize.split()))


if __name__ == '__main__':
    run_conceptual_maps()