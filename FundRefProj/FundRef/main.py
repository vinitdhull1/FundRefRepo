from FundRefClass import dataPreparation
from os.path import basename
from os import path
from django.conf import settings

obj = dataPreparation("D:\\FundRefProjDir\\FundRefProj\\FundRef\\training_data-26102021.csv")

obj.dataCreation()