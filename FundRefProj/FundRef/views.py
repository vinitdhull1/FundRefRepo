from __future__ import unicode_literals, print_function
from django.http.response import JsonResponse
from django.shortcuts import render
from rest_framework.decorators import api_view
from os.path import basename
from os import path
from django.conf import settings
from pathlib import Path
import spacy
from tqdm import tqdm
import pandas as pd
from flask import Flask,jsonify,request,make_response,url_for,redirect
import requests, json

app = Flask(__name__)


#NER model path
output_dir=Path(r"/ner")

@api_view(['GET','POST'])
def file_loc(request):
    if request.method == 'POST':
        # file_data = request.get_json()
        file_data = json.loads(request.body)
        text = file_data["text"]
        text = text.replace(";","")
        nlp2 = spacy.load(output_dir)

        doc = nlp2(text)
        output = []

        with open('doi_ins.json') as f:
            doi_ii = json.load(f)
        for ent in doc.ents:
            print(ent.text, ent.label_)
            if ent.label_ == "institution":
                x = ent.text
                y = ''
                for key, value in doi_ii.items():
                    if x in value:
                        y = str(key)
                output.append({ent.label_: ent.text, "doi": y})
            elif ent.label_ == "award-id":
                output.append({ent.label_: ent.text})
                # x = ent.text
                # x = x.split(" ")
                # y=""
                # for i in x:
                #     if not i.isalpha():
                #         y=i
                # output.append({ent.label_: y})

    return JsonResponse({"output": output})

