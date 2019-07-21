import random as rn
import numpy as np
import sys
import json

__all__ = ['DataGen']

np.random.seed(1)
rn.seed(1)

class DataGen():
    maxSequenceLength = 25
    maxColors = 4
    maxAnimals = 4
    countsByLabel = {}

    def shuffleThis(self,listIn):
        listOut = listIn.copy()
        rn.shuffle(listOut)
        return listOut

    def getData(self):
        with open ('words.txt' , 'r') as f:
            words = sorted(list(set(f.read().lower().strip().split(','))))
        X, labels_color, labels_animal = [], [], []
        colorNames = ['violet', 'indigo', 'blue', 'green', 'yellow', 'orange', 'red']
        animalNames = ['fox', 'parrot', 'bunny', 'dog', 'cat', 'lion', 'tiger', 'bear']
        nWords = len(words) + len(colorNames) + len(animalNames)
        colorNames = colorNames + ['none']
        animalNames = animalNames + ['none']
        name2label_colors = dict(zip(colorNames,range(0,len(colorNames))))
        name2label_animals = dict(zip(animalNames,range(0,len(animalNames))))

        print ("Vocab Size:", nWords)
        dummyWord = "zzzzz"

        for label in colorNames:
            self.countsByLabel[label] = 0
        for label in animalNames:
            self.countsByLabel[label] = 0

        for i in range(1000):
            sequenceLength = np.random.randint(low=15,high=self.maxSequenceLength+1, size=1)[0]
            label_color = np.zeros(len(colorNames), dtype=int)
            label_animal = np.zeros(len(animalNames), dtype=int)
            nColors = np.random.randint(low=1,high=self.maxColors +1, size=1)[0]
            nAnimals = np.random.randint(low=1,high=self.maxAnimals +1, size=1)[0]
            colors = rn.sample(colorNames[0:-1], nColors)
            animals = rn.sample(animalNames[0:-1], nAnimals)

            for color in colors:
                label_color[name2label_colors[color]] = 1
                self.countsByLabel[color] = self.countsByLabel[color] + 1
            for animal in animals:
                label_animal[name2label_animals[animal]] = 1
                self.countsByLabel[animal] = self.countsByLabel[animal] + 1

            doc = self.shuffleThis(rn.sample(words, sequenceLength - nColors - nAnimals) + colors + animals)
            doc = doc + [dummyWord]*(self.maxSequenceLength - sequenceLength)
            X.append(doc)
            labels_color.append(label_color.tolist())
            labels_animal.append(label_animal.tolist())

        noneLabel_color = np.zeros(len(colorNames), dtype=int).tolist()
        noneLabel_color[-1] = 1
        noneLabel_animal = np.zeros(len(animalNames), dtype=int).tolist()
        noneLabel_animal[-1] = 1
        for i in range(300):
            sequenceLength = np.random.randint(low=15,high=self.maxSequenceLength+1, size=1)[0]
            doc = rn.sample(words, sequenceLength)
            doc = doc + [dummyWord]*(self.maxSequenceLength - sequenceLength)
            X.append(doc)
            labels_color.append(noneLabel_color)
            labels_animal.append(noneLabel_animal)
            self.countsByLabel['none'] = self.countsByLabel['none'] + 1

        print ('Counts by Label:',self.countsByLabel)
        return X, labels_color, labels_animal, colorNames, animalNames, self.maxSequenceLength

X, labels_color, labels_animal, colorNames, animalNames, seqL  = DataGen().getData()
result = {'X': X, 'labels_color' : labels_color, 'labels_animal' : labels_animal, 'colorNames' : colorNames, 'animalNames' : animalNames, 'sequenceLength' : seqL}

f = open ('data/data.json','w')
out = json.dumps(result, ensure_ascii=True)
f.write(out)
f.close()

