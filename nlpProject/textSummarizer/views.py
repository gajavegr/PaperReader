from lib2to3.pgen2 import token
import os
from django.shortcuts import render

import nltk
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx

import io

import requests
import PyPDF2
from . import data_func

import urllib.request

# def read_article(file_name):
def read_article(file_string):
    # file = open(file_name, "r")
    # filedata = file.readlines()
    filedata = file_string
    # article = filedata[0].split(". ")
    article = filedata.split(". ")
    sentences = []

    for sentence in article:
        print(sentence)
        sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" "))
    sentences.pop() 
    
    return sentences

def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []
 
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]
 
    all_words = list(set(sent1 + sent2))
 
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
 
    # build the vector for the first sentence
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1
 
    # build the vector for the second sentence
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1
 
    return 1 - cosine_distance(vector1, vector2)
 
def build_similarity_matrix(sentences, stop_words):
    # Create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
 
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2: #ignore if both are same sentences
                continue 
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

    return similarity_matrix


# def generate_summary(file_name, top_n=5):
def generate_summary(file_string, top_n=5):
    # file_string = file_string.replace('"','\\"')
    nltk.download("stopwords")
    stop_words = stopwords.words('english')
    summarize_text = []

    # Step 1 - Read text anc split it
    # sentences =  read_article(file_name)
    sentences =  read_article(file_string)

    # Step 2 - Generate Similary Martix across sentences
    sentence_similarity_martix = build_similarity_matrix(sentences, stop_words)

    # Step 3 - Rank sentences in similarity martix
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
    scores = nx.pagerank(sentence_similarity_graph)

    # Step 4 - Sort the rank and pick top sentences
    ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)    
    print("Indexes of top ranked_sentence order are ", ranked_sentence)    

    for i in range(top_n):
      summarize_text.append(" ".join(ranked_sentence[i][1]))

    # Step 5 - Offcourse, output the summarize text
    # print("Summarize Text: \n", ". ".join(summarize_text))
    summarized_text = ". ".join(summarize_text)
    print("Summarize Text: \n", summarized_text)
    return summarized_text

# Create your views here.
def home(request):
    return render(request,"home.html",{})

def regex(request):
    import re
    if request.method == "POST":
        phonepattern=r"\d{3}-\d{3}-\d{4}"
        regex_area = request.POST["regexform"]
        # regex_phone = re.findall(phonepattern,regex_area)
        regex_phone = generate_summary(regex_area,2)
        return render(request, "regex.html",{"phone":regex_phone})
    else:
        return render(request,"home.html",{})

def lemma(request):
    # # import nltk
    # # # nltk.download("omw-1.4")
    # # from nltk.stem import WordNetLemmatizer
    # # wordnet_lemmatizer = WordNetLemmatizer()
    # # if request.method=="POST":
    # #     lemma_area = request.POST["lemmaform"]
    # #     tokenization = nltk.word_tokenize(lemma_area)
    # #     tokens = [i for i in tokenization]
    # #     lemma_list = [wordnet_lemmatizer.lemmatize(i) for i in tokens]
    # #     lemmas = [i for i in lemma_list]
    # #     return render(request,"lemma.html", {"unprocessed":tokens,"lemmatized":lemmas})
    # # else:
    # #     return render(request,"home.html",{})
    
    # import spacy
    # nlp = spacy.load("en_core_web_sm")
    # if request.method == "POST":
    #     lemma_area = request.POST["lemmaform"]
    #     lemma_area = nlp(lemma_area)
    #     token_list = [i.text for i in lemma_area]
    #     tokens = [i for i in token_list]
    #     lemmas_list = [i.lemma_ for i in lemma_area]
    #     lemmas = [i for i in lemmas_list]
    #     return render(request,"lemma.html",{"unprocessed":tokens,"lemmatized":lemmas})
    # else:
    #     return render(request,"home.html",{})
    if request.method == "POST":
        url = request.POST["lemmaform"]
        # url = 'http://www.arkansasrazorbacks.com/wp-content/uploads/2017/02/Miami-Ohio-Game-2.pdf'
        r = requests.get(url)
        # f = io.BytesIO(r.content)
        pdf_filename = "downloaded"
        download_file(url,pdf_filename)
        f = f"{pdf_filename}.pdf"
        content = data_func.convert_pdf_to_string(f)
        # print(content)
        os.remove(f)
        # return render(request,"lemma.html",{"unprocessed":tokens,"lemmatized":lemmas})
        return render(request,"lemma.html",{"lemmatized":generate_summary(content,2)})
    else:
        return render(request,"home.html",{})


def download_file(download_url, filename):
    response = urllib.request.urlopen(download_url)    
    file = open(filename + ".pdf", 'wb')
    file.write(response.read())
    file.close()



def pos(request):
    return render(request,"pos.html",{})

def ner(request):
    return render(request,"ner.html",{})


