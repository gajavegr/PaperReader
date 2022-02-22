import os
import re
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
        # print(sentence)
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
    # print("Indexes of top ranked_sentence order are ", ranked_sentence)    

    for i in range(top_n):
      summarize_text.append(" ".join(ranked_sentence[i][1]))

    # Step 5 - Offcourse, output the summarize text
    # print("Summarize Text: \n", ". ".join(summarize_text))
    summarized_text = ". ".join(summarize_text)
    # print("Summarize Text: \n", summarized_text)
    return summarized_text

# Create your views here.
def home(request):
    return render(request,"home.html",{})

def summarizeText(request):
    if request.method == "POST":
        textToSummarize = request.POST["summarizeForm"]
        generatedSummary = generate_summary(textToSummarize,2)
        return render(request, "textSummary.html",{"text_summary":generatedSummary})
    else:
        return render(request,"home.html",{})

def summarizeAbstract(request):
    if request.method == "POST":
        url = request.POST["abstractSummaryForm"]
        # url = 'http://www.arkansasrazorbacks.com/wp-content/uploads/2017/02/Miami-Ohio-Game-2.pdf'
        r = requests.get(url)
        # f = io.BytesIO(r.content)
        pdf_filename = "downloaded"
        download_file(url,pdf_filename)
        f = f"{pdf_filename}.pdf"
        content = data_func.convert_pdf_to_string(f)
        # print(content)
        abstract_location = 0
        introduction_location = 0
        if content.__contains__("Abstract"):
            # print("found Abstract")
            abstract_location = content.index("Abstract")
        if content.__contains__("Introduction"):
            # print("found Introduction")
            introduction_location = content.index("Introduction")
        abstract = content[abstract_location+8:introduction_location-5].replace("\n"," ").replace(" - ","")
        # pattern = "Abstract(.*)Introduction"
        # abstract = re.search(pattern, content).group(1)
        # print(abstract)
        os.remove(f)
        return render(request,"summarizeAbstract.html",{"paper":url,"abstract":abstract,"summarizedAbstract":generate_summary(abstract,2)})
    else:
        return render(request,"home.html",{})


def download_file(download_url, filename):
    response = urllib.request.urlopen(download_url)    
    file = open(filename + ".pdf", 'wb')
    file.write(response.read())
    file.close()
