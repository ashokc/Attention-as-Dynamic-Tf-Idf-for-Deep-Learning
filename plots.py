import numpy as np
import glob, os, re, sys, json
import matplotlib.pyplot as plt
from PIL import Image

#get_ipython().magic('matplotlib inline')

def plotBars (allResults, objective, sentenceIndex):
    width = 0.05
    labelColors = ['violet', 'indigo', 'blue', 'green', 'yellow', 'orange', 'red','none'] 
    labelAnimals = ['fox', 'parrot', 'bunny', 'dog', 'cat', 'lion', 'tiger', 'bear','none']

# plot probability
    if (objective == 'colors'):
        xLabels = labelColors
        colors = labelColors[0:-1]
    elif (objective == 'animals'):
        xLabels = labelAnimals
        colors = None
    xVals = np.arange(len(xLabels))
    for attention in ['yes', 'no']:
        probabilities = allResults[objective][attention]['predictions'][sentenceIndex]['predictedProbabilities']
        name = './results/' + str(sentenceIndex) + '-' + objective + '-probability-' + attention + '.png'
        plotFig(xVals, xLabels, probabilities, name, colors)

# plot attention weights
    for attention in ['yes', 'no']:
        words = allResults[objective][attention]['predictions'][sentenceIndex]['sentence']
        xVals = np.arange(len(words))
        colors = []
        for i,word in enumerate(words):
            if (word in labelColors):
                colors.append(word)
            else:
                colors.append('k')
        if attention == 'yes':
            attentionWeights = allResults[objective][attention]['predictions'][sentenceIndex]['attentionWeights']
            name = './results/' + str(sentenceIndex) + '-' + objective + '-attentionWeights-' + attention + '.png'
            plotFig(xVals, words, attentionWeights, name, colors )

def plotFig (xVals, xLabels, heights, name, colors=None):
    fig, ax = plt.subplots()
    if colors:
        plt.bar(xVals, heights, color=colors)
    else:
        plt.bar(xVals, heights)
    plt.xticks(xVals, xLabels)
    plt.xticks(rotation=90)
    plt.tight_layout()
    fig.savefig(name, format='png', dpi=720)

def main(sentenceIndex):
    allResults = {}
    for objective in ['colors', 'animals']:
        allResults[objective] = {}
        for attention in ['yes', 'no']:
            filename = './results/' + attention + '-' + objective + '.json'
            with open (filename) as fh:
                result = json.loads(fh.read())
                allResults[objective][attention] = result
    plotBars (allResults, 'colors', sentenceIndex)
    plotBars (allResults, 'animals', sentenceIndex)

args = sys.argv
if (len(args) < 1):
    logger.critical ("Need 1 arg... testSentenceIndex  Exiting")
    sys.exit(0)
else:
    sentenceIndex = int(args[1])

if __name__ == '__main__':
    main(sentenceIndex)

