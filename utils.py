#import libs
import json

def readJsonFile(path="./dataset/tmp/ego4d.json"):
    with open(path) as jsonFile:
        jsonObject = json.load(jsonFile)
        jsonFile.close()
        return jsonObject