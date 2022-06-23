from math import ceil
from time import time
import spacy
# from pytextrank import TextRank
from collections import Counter
from string import punctuation



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
@click.option('--data', '-d', default='data/input_angelo.txt', required=False, show_default=True,
              help=u'Input text file route')
def run_conceptual_maps(data):

    dict_idNodes_nodes = {}
    dict_idNodes_relations = {}

    # Title of the map (ask to user?)
    origin_name = 'ORIGIN'
    dict_idNodes_nodes['ORIGIN'] = origin_name
    dict_idNodes_relations['ORIGIN'] = [[],'']

    f = open(data)

    sections = f.readlines()

    # print(input)

    nltk.download('punkt')

    i_sec = 0
    for sec in sections:


        # Link origin with the section node
        title_node_id = 's_'+str(i_sec)
        dict_idNodes_relations['ORIGIN'][0].append(title_node_id)

        detected_lang = detect(sec)

        sentences = split_text(sec, detected_lang)

        n_sentences = len(sentences)-1

        # The title of the section should be the first word in the list 
        sec_title = sentences[0]
        
        print('--------',sec_title,'--------')

        translations = []

        # Splitting the text into sentences we ensure not reaching the character limit of Google Translator
        print('Translating text into English...')
        tic = time()
        
        # If we would work with so long sections we should translate sentence by sentence
        # for sent in sentences[1:]:  # [1:] because we don't need to translate the title
        #     translations.append(GoogleTranslator(
        #         source='auto', target='en').translate(sent))

        # full_translation = " ".join(
        #     w for word in translations for w in word.split())

        # However it seems that with the current idea sections are not so long, and translating th whole text is
        # quite faster

        sec_text = " ".join(w for word in sentences[1:] for w in word.split())
        full_translation = GoogleTranslator(source='auto', target='en').translate(sec)

        print('Translation time: {} s'.format(round(time()-tic,3)))

        tic = time()

        resolved_doc = full_translation
        print('Summarizing text...')

        p = 0.3
        # Summarizing to a 30% of the original
        n_summary_sentences = ceil(p*n_sentences)
        # summarize = top_sentence(resolved_doc,n_summary_sentences)
        summarize = top_sentence(resolved_doc,5)


        # 5 sentences to summarize
        # summarize = top_sentence(resolved_doc,5)


        print('Summarization time: {} s'.format(round(time()-tic,3)))

        # Splitting English summarized text
        en_sentences = split_text(summarize, 'en')
        
        # Filling dictionaries for composing the map
        dict_idNodes_nodes, dict_idNodes_relations = obtaining_nodes_relations(
            sec_title, i_sec, en_sentences, dict_idNodes_nodes, dict_idNodes_relations, detected_lang)

        i_sec += 1

    # Generating an output to check nodes and relations are correct
    generate_simple_map(dict_idNodes_nodes, dict_idNodes_relations)


def split_text(text, lang):
    if lang == 'es':
        tokenizer = nltk.data.load('tokenizers/punkt/spanish.pickle')
    elif lang == 'it':
        tokenizer = nltk.data.load('tokenizers/punkt/italian.pickle')
    elif lang == 'en':
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    elif lang == 'fr':
        tokenizer = nltk.data.load('tokenizers/punkt/french.pickle')

    sentences = tokenizer.tokenize(text)

    return sentences


def generate_simple_map(dict_idNodes_nodes, dict_idNodes_relations):
    for k, v in dict_idNodes_relations.items():
        for i in range(len(v[0])):
            if str(dict_idNodes_nodes[v[0][i]]) != "":
                node_src = str(dict_idNodes_nodes[k])
                relation_name = str(v[1])
                node_dst = str(dict_idNodes_nodes[v[0][i]])
                print('\n', node_src, '--[', relation_name, ']-->', node_dst)


def obtaining_nodes_relations(sec_title, i_sec, en_sentences, dict_idNodes_nodes, dict_idNodes_relations, lan):
    dict_nodes_idNodes = {}

    id = 0
    id_relations = 0

    # Create the source node of the section.
    title_node_id = 's_'+str(i_sec)
    dict_nodes_idNodes[sec_title] = title_node_id
    dict_idNodes_relations[title_node_id] = [[],'']
    
    nlp = spacy.load('en_core_web_trf')

    for sentence in en_sentences:
        doc = nlp(sentence)

        # Each new subject will be a new node
        subject_phrase_en = get_subject_phrase(doc)
        if subject_phrase_en == "":
            # if there is not subject we skip this sentence for now
            continue
        subject_phrase = GoogleTranslator(source='auto', target=lan).translate(subject_phrase_en)

        # the same subject could be in different sentences, but we should be keep in
        # mind that it would still be the same node but with several relationships
        if not subject_phrase in dict_nodes_idNodes:
            new_node_id = title_node_id+'_n_'+str(id)
            # Add the node to the node dict
            dict_nodes_idNodes[subject_phrase] = new_node_id
            # Connect the new node with the source section node
            dict_idNodes_relations[title_node_id][0].append(new_node_id)
            id += 1
        
        # The relation will be composed by the verb
        root_en = get_verb_and_auxs(doc)
        
        # The second node obtained from the sentence would be composed by the rest of
        # the sentence meaning (not exactly the predicate)
        # object_phrase_en = get_predicate(doc)
        object_phrase_en = get_rest_phrase(doc, subject_phrase_en, root_en)
        object_phrase = GoogleTranslator(source='auto', target=lan).translate(object_phrase_en)

        if not object_phrase in dict_nodes_idNodes:
            dict_nodes_idNodes[object_phrase] = title_node_id+'_n_'+str(id)
            id += 1

        if not (dict_nodes_idNodes[subject_phrase] in dict_idNodes_relations):
            dict_idNodes_relations[dict_nodes_idNodes[subject_phrase]] = [[],'']

        dict_idNodes_relations[dict_nodes_idNodes[subject_phrase]][0].append(
            dict_nodes_idNodes[object_phrase])

        
        # dict_idRealtions_relations[(
        #     dict_nodes_idNodes[subject_phrase], dict_nodes_idNodes[object_phrase])] = root
        root = GoogleTranslator(source='auto', target=lan).translate(root_en)
        dict_idNodes_relations[dict_nodes_idNodes[subject_phrase]][1] = root

        id_relations += 1

    # Change values for keys and stores in a new dictionary
    for k, v in dict_nodes_idNodes.items():
        dict_idNodes_nodes[v] = k

    return dict_idNodes_nodes, dict_idNodes_relations


# Methods to obtain different parts of the sentence
# Extracting the sentence subjects
def get_subject_phrase(doc):
    returned_subjects = []
    for token in doc:
        if ("subj" in token.dep_):
            subtree = list(token.subtree)
            start = subtree[0].i
            end = subtree[-1].i + 1
            return str(doc[start:end])
    return ""

# Extracting the "predicate" (not exactly) from a sentence
def get_predicate(doc):
    for token in doc:
        # print(token, token.dep_)
        if ("ROOT" in token.dep_):
            return doc[token.i + 1: -1].text
    
    return ""

def get_object_phrase(doc):
    for token in doc:
        print(token, token.dep_)
        if ("dobj" in token.dep_):
            subtree = list(token.subtree)
            start = subtree[0].i
            end = subtree[-1].i + 1
            return str(doc[start:end])


def get_rest_phrase(doc,subj,verb):
    sentence = doc.text
    sentence = sentence.replace(subj,"")
    sentence = sentence.replace(verb,"")

    return sentence

# Extracting the sentence verb
def get_verb_and_auxs(doc):
    neg = ""
    aux = ""
    res = ""
    for token in doc:
        # print(token, token.dep_)
        if ("ROOT" in token.dep_):
            for i in token.children:
                if(i.dep_ == "aux"):
                    # print("child aux:",i)
                    aux = i
                elif(i.dep_ == "neg"):
                    neg = i

            # There is only a root so return directly
            return (str(aux)+" "+str(neg)+" "+token.text).lstrip()#, token.i
    return ""


def print_token_dependences(doc):
    for token in doc:
        print(token.text, token.dep_, token.pos)

def top_sentence(text, limit):
    nlp = spacy.load('en_core_web_trf')
    keyword = []
    pos_tag = ['PROPN', 'ADJ', 'NOUN', 'VERB']
    # doc = nlp(text.lower())
    doc = nlp(text)
    for token in doc:
        if(token.text in nlp.Defaults.stop_words or token.text in punctuation):
            continue
        if(token.pos_ in pos_tag):
            keyword.append(token.text)
    
    freq_word = Counter(keyword)
    max_freq = Counter(keyword).most_common(1)[0][1]
    for w in freq_word:
        freq_word[w] = (freq_word[w]/max_freq)
        
    sent_strength={}
    for sent in doc.sents:
        for word in sent:
            if word.text in freq_word.keys():
                if sent in sent_strength.keys():
                    sent_strength[sent]+=freq_word[word.text]
                else:
                    sent_strength[sent]=freq_word[word.text]
    
    summary = []
    
    sorted_x = sorted(sent_strength.items(), key=lambda kv: kv[1], reverse=True)
    
    counter = 0
    for i in range(len(sorted_x)):
        summary.append(str(sorted_x[i][0]).capitalize())

        counter += 1
        if(counter >= limit):
            break
            
    return ' '.join(summary)


if __name__ == '__main__':
    run_conceptual_maps()
