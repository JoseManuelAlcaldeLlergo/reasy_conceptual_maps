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
@click.option('--data', '-d', default='data/input.txt', required=False, show_default=True,
              help=u'Input text file route')
def run_conceptual_maps(data):

    dict_idNodes_nodes = {}
    dict_idNodes_relations = {}
    dict_idRealtions_relations = {}

    # Title of the map (ask to user?)
    origin_name = 'ORIGIN'
    dict_idNodes_nodes['ORIGIN'] = origin_name
    dict_idNodes_relations['ORIGIN'] = []

    f = open(data)

    sections = f.readlines()

    # print(input)

    nltk.download('punkt')

    i_sec = 0
    for sec in sections:


        # Link origin with the section node
        title_node_id = 's_'+str(i_sec)
        dict_idNodes_relations['ORIGIN'].append(title_node_id)
        dict_idRealtions_relations[('ORIGIN', title_node_id)] = ''

        detected_lang = detect(sec)

        sentences = split_text(sec, detected_lang)

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
        # bert_summarizer = Summarizer()

        resolved_doc = full_translation
        # print(len(resolved_doc.split()))
        print('Summarizing text...')
        # Summarizing to a 20% of the original
        # summarize = bert_summarizer(resolved_doc, ratio=0.2)
        # summarize = summarize_text(resolved_doc)
        summarize = top_sentence(resolved_doc,5)


        print('Summarization time: {} s'.format(round(time()-tic,3)))

        # Splitting English summarized text
        en_sentences = split_text(summarize, 'en')
        
        # Filling dictionaries for composing the map
        dict_idNodes_nodes, dict_idNodes_relations, dict_idRealtions_relations = obtaining_nodes_relations(
            sec_title, i_sec, en_sentences, dict_idNodes_nodes, dict_idNodes_relations, dict_idRealtions_relations)

        i_sec += 1

    # Generating an output to check nodes and relations are correct
    generate_simple_map(detected_lang, dict_idNodes_nodes,
                        dict_idNodes_relations, dict_idRealtions_relations)


def summarize_text(text):
    nlp = spacy.load('en_core_web_trf')

    tr = TextRank()
    nlp.add_pipe(tr.PipelineComponent, name="textrank", last=True)

    doc = nlp(text)

    sent_bounds = [ [s.start, s.end, set([])] for s in doc.sents ]

    limit_phrases = 4

    phrase_id = 0
    unit_vector = []

    for p in doc._.phrases:
        print(phrase_id, p.text, p.rank)
        
        unit_vector.append(p.rank)
        
        for chunk in p.chunks:
            print(" ", chunk.start, chunk.end)
            
            for sent_start, sent_end, sent_vector in sent_bounds:
                if chunk.start >= sent_start and chunk.start <= sent_end:
                    print(" ", sent_start, chunk.start, chunk.end, sent_end)
                    sent_vector.add(phrase_id)
                    break

        phrase_id += 1

        if phrase_id == limit_phrases:
            break

    for sent in doc.sents:
        print(sent)
    return text

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


def generate_simple_map(detected_lang, dict_idNodes_nodes, dict_idNodes_relations, dict_idRealtions_relations):
    for k, v in dict_idNodes_relations.items():
        for i in range(len(v)):
            if str(dict_idNodes_nodes[v[i]]) != "":
                node_src = str(dict_idNodes_nodes[k])
                relation_name = str(dict_idRealtions_relations[(k, v[i])])
                node_dst = str(dict_idNodes_nodes[v[i]])
                print('\n', GoogleTranslator(source='auto', target=detected_lang).translate(node_src), '--[', GoogleTranslator(
                    source='auto', target=detected_lang).translate(relation_name), ']-->', GoogleTranslator(source='auto', target=detected_lang).translate(node_dst))


def obtaining_nodes_relations(sec_title, i_sec, en_sentences, dict_idNodes_nodes, dict_idNodes_relations, dict_idRealtions_relations):
    dict_nodes_idNodes = {}

    id = 0
    id_relations = 0

    # Create the source node of the section.
    title_node_id = 's_'+str(i_sec)
    dict_nodes_idNodes[sec_title] = title_node_id
    dict_idNodes_relations[title_node_id] = []
    
    nlp = spacy.load('en_core_web_trf')

    for sentence in en_sentences:
        doc = nlp(sentence)

        # Each new subject will be a new node
        subject_phrase = get_subject_phrase(doc)

        # the same subject could be in different sentences, but we should be keep in
        # mind that it would still be the same node but with several relationships
        if not subject_phrase in dict_nodes_idNodes:
            new_node_id = title_node_id+'_n_'+str(id)
            # Add the node to the node dict
            dict_nodes_idNodes[subject_phrase] = new_node_id
            # Connect the new node with the source section node
            dict_idNodes_relations[title_node_id].append(new_node_id)
            # Naming the relation as ''
            dict_idRealtions_relations[(title_node_id, new_node_id)] = ''
            id += 1

        # The second node obtained from the sentence would be composed by the rest of
        # the sentence meaning (not exactly the predicate)
        object_phrase = get_predicate(doc)
        if not object_phrase in dict_nodes_idNodes:
            dict_nodes_idNodes[object_phrase] = title_node_id+'_n_'+str(id)
            id += 1

        if not (dict_nodes_idNodes[subject_phrase] in dict_idNodes_relations):
            dict_idNodes_relations[dict_nodes_idNodes[subject_phrase]] = []

        dict_idNodes_relations[dict_nodes_idNodes[subject_phrase]].append(
            dict_nodes_idNodes[object_phrase])
        current_rel = "r_"+str(id_relations)

        root, root_position_start = get_verb_and_auxs(doc)
        dict_idRealtions_relations[(
            dict_nodes_idNodes[subject_phrase], dict_nodes_idNodes[object_phrase])] = root

        id_relations += 1

    # Change values for keys and stores in a new dictionary
    for k, v in dict_nodes_idNodes.items():
        dict_idNodes_nodes[v] = k

    return dict_idNodes_nodes, dict_idNodes_relations, dict_idRealtions_relations


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
    return None

# Extracting the sentence objects


def get_object_phrase(doc):
    for token in doc:
        print(token, token.dep_)
        if ("dobj" in token.dep_):
            subtree = list(token.subtree)
            start = subtree[0].i
            end = subtree[-1].i + 1
            return str(doc[start:end])

# Extracting the sentence verb


def get_root_phrase(doc):
    res = ""
    for token in doc:
        # print(token, token.dep_)
        if ("ROOT" in token.dep_):
            # There is only a root so return directly
            return token.text, token.i


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
            return (str(aux)+" "+str(neg)+" "+token.text).lstrip(), token.i

# Extracting the "predicate" (not exactly) from a sentence


def get_predicate(doc):
    res = ""
    for token in doc:
        # print(token, token.dep_)
        if ("ROOT" in token.dep_):
            # There is only a root so return directly
            return doc[token.i + 1: -1]


def get_prepositional_phrase_objs(doc):
    prep_spans = []
    for token in doc:
        if ("pobj" in token.dep_):
            subtree = list(token.subtree)
            start = subtree[0].i
            end = subtree[-1].i + 1
            prep_spans.append(doc[start:end])
    return prep_spans


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
