from __future__ import unicode_literals, print_function
import plac
import random
from pathlib import Path
from os.path import basename
from os import path
from django.conf import settings
import spacy
from tqdm import tqdm
import pandas as pd
import ast

class dataPreparation:

    def __init__(self, dataPath):
        self.dataPath = dataPath

    #driver method for data creation for NER
    def dataCreation(self):
        #data = pd.read_csv("training_data.csv")
        file = r"{}".format(self.dataPath)
        data = pd.read_csv(file)
        data.dropna(inplace=True)

        train_list = []

        # prepare training data for NER
        for i, j in zip(data.Text, data.Entities):
            train_list.append((i, {'entities':ast.literal_eval(j)}))
            #print(i, j)

        model = None
        #define path for model
        output_dir = Path(r"/ner")
        n_iter = 100

        # load the model

        if model is not None:
            nlp = spacy.load(model)
            print("Loaded model '%s'" % model)
        else:
            nlp = spacy.blank('en')
            print("Created blank 'en' model")

        # set up the pipeline

        if 'ner' not in nlp.pipe_names:
            ner = nlp.create_pipe('ner')
            nlp.add_pipe(ner, last=True)
        else:
            ner = nlp.get_pipe('ner')

        for _, annotations in train_list:
            for ent in annotations.get('entities'):
                ner.add_label(ent[2])

        other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
        with nlp.disable_pipes(*other_pipes):  # only train NER
            optimizer = nlp.begin_training()
            for itn in range(n_iter):
                random.shuffle(train_list)
                losses = {}
                for text, annotations in tqdm(train_list):
                    nlp.update(
                        [text],
                        [annotations],
                        drop=0.5,
                        sgd=optimizer,
                        losses=losses)
                print(losses)

        #save model to defined path
        if output_dir is not None:
            output_dir = Path(output_dir)
            if not output_dir.exists():
                output_dir.mkdir()
            nlp.to_disk(output_dir)
            print("Saved model to", output_dir)
