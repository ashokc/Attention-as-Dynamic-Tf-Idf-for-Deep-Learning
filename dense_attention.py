import random as rn
import numpy as np
import keras
import os
import json
from sklearn.model_selection import train_test_split
import tensorflow as tf
import sys
from keras.utils import plot_model
from attention_layer import Attention
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
#All this for reproducibility
np.random.seed(1)
rn.seed(1)
tf.set_random_seed(1)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
keras.backend.set_session(sess)

args = sys.argv
if (len(args) < 3):
    logger.critical ("Need 3 args... attention:'yes'/'no' & objective:'colors'/'animals' & mask: 'yes'/'no'  Exiting")
    sys.exit(0)
else:
    attention = args[1]
    objective = args[2]
    mask = args[3]
    if (mask == "yes"):
        mask_zero = True
    elif (mask == "no"):
        mask_zero = False

f = open ("./data/data.json",'r')
dataIn = json.loads(f.read())
f.close()
X, labels_color, labels_animal, colorNames, animalNames, sequenceLength = dataIn['X'], np.array(dataIn['labels_color']), np.array(dataIn['labels_animal']), dataIn['colorNames'], dataIn['animalNames'], dataIn['sequenceLength']

if mask_zero:
    newX = []
    for doc in X:
        Xsmall = [word for word in doc if word != 'zzzzz']
        newX.append(Xsmall)
    X = newX.copy()

if (objective == 'colors'):
    names = colorNames
    labels = labels_color
elif (objective == 'animals'):
    names = animalNames
    labels = labels_animal

wordVectorLength = 128
denseUnits = 64
N_a = 100
epochs = 200

# get encoded & padded documents
kTokenizer = keras.preprocessing.text.Tokenizer()
kTokenizer.fit_on_texts(X)
encoded_docs = kTokenizer.texts_to_sequences(X)
index_word = {value: key for key, value in kTokenizer.word_index.items()}
if mask_zero:
    Xencoded = keras.preprocessing.sequence.pad_sequences(encoded_docs, maxlen=sequenceLength, padding='post')
else:
    Xencoded = np.array([np.array(xi) for xi in kTokenizer.texts_to_sequences(X)])
print ('Vocab:', len(index_word), 'Xencoded:', Xencoded.shape, 'Labels:', labels.shape)

def applyAttention (wordVectorRowsInSentence):   # [*, N_s, N_f] N_f = wordVector size
    N_f = wordVectorRowsInSentence.shape[-1]
    uiVectorRowsInSentence = keras.layers.Dense(units=N_a, activation='tanh')(wordVectorRowsInSentence) # [*, N_s, N_a]
    vVectorColumnMatrix = keras.layers.Dense(units=1, activation='tanh')(uiVectorRowsInSentence) # [*, N_s, 1]
    vVector = keras.layers.Flatten()(vVectorColumnMatrix)    # [*, N_s]
    attentionWeightsVector = keras.layers.Activation('softmax', name='attention_vector_layer')(vVector) # [*,N_s]
    attentionWeightsMatrix = keras.layers.RepeatVector(N_f)(attentionWeightsVector)   # [*,N_f, N_s]
    attentionWeightRowsInSentence = keras.layers.Permute([2, 1])(attentionWeightsMatrix)  # [*,N_s, N_f]
    attentionWeightedSequenceVectors = keras.layers.Multiply()([wordVectorRowsInSentence, attentionWeightRowsInSentence])  # [*,N_s, N_f]
    attentionWeightedSentenceVector = keras.layers.Lambda(lambda x: keras.backend.sum(x, axis=1), output_shape=lambda s: (s[0], s[2]))(attentionWeightedSequenceVectors)    # [*,N_f]
    return attentionWeightedSentenceVector

def getModel():
    listOfWords = keras.layers.Input((sequenceLength,), dtype="int32")
    embed = keras.layers.Embedding(input_dim=len(kTokenizer.word_index)+1, output_dim=wordVectorLength, input_length=sequenceLength, trainable=True,mask_zero=mask_zero)(listOfWords)
    if (attention == 'yes'):
        if mask_zero:
            print ('Mask Zero:' , mask_zero, ' : Using the custom Attention Layer from Christos Baziotis') 
            vectorsForPrediction, attention_vectors = Attention(return_attention=True, name='attention_vector_layer')(embed)
        else:
            print ('Mask Zero:' , mask_zero, ' : Using the function described here with repeat & permute blocks...')
            vectorsForPrediction = applyAttention(embed)
    elif (attention == 'no'):
        countDocVector = keras.layers.Lambda(lambda x: keras.backend.sum(x, axis=1), output_shape=lambda s: (s[0], s[2]))(embed)
        vectorsForPrediction = keras.layers.Dense(units=denseUnits, activation='relu')(countDocVector)
    predictions = keras.layers.Dense(len(names), activation='sigmoid',use_bias=False)(vectorsForPrediction)
    model = keras.models.Model(inputs=listOfWords, outputs=predictions)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['categorical_accuracy'])
    print(model.summary())
    plot_model(model, show_shapes=True, to_file='results/' + attention + '-' + mask + '.png')
    if (attention == 'yes'):
        attention_layer_model = keras.models.Model(inputs=model.input, outputs=model.get_layer('attention_vector_layer').output)
    else:
        attention_layer_model = None
    return model, attention_layer_model

# Train/Valid/Test Split

train_x, test_x, train_labels, test_labels = train_test_split(Xencoded, labels, test_size=0.2, random_state=1)
train_x_1, valid_x, train_labels_1, valid_labels = train_test_split(train_x, train_labels, test_size=0.2, random_state=1)

print ('Train/Valid/Test:', len(train_labels_1), len(valid_labels), len(test_labels))

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1.0e-12, patience=15, verbose=2, mode='auto', restore_best_weights=True)
np.set_printoptions(precision=4)

result = {}
model, attention_layer_model = getModel()
history = model.fit(x=train_x_1, y=train_labels_1, epochs=epochs, batch_size=64, shuffle=True, validation_data = (valid_x, valid_labels), verbose=2, callbacks=[early_stop])
result['history'] = history.history
result['test_loss'], result['test_accuracy'] = model.evaluate(test_x, test_labels, verbose=2)
predicted = model.predict(test_x, verbose=2)
if (attention == 'yes' and not mask_zero):
    attention_vectors = attention_layer_model.predict(test_x, verbose=2)
if (attention == 'yes' and mask_zero):
    tmp, attention_vectors = attention_layer_model.predict(test_x, verbose=2)

print ('Testing the predictions:')

predResults = []
names = np.array(names)
totalError = np.zeros_like(test_labels[0])
for j, sample in enumerate(test_x):
    predResult = {}
    sentence = [index_word[k] for k in sample if k > 0]
    if (attention == 'yes'):
        predResult['attentionWeights'] = attention_vectors[j].tolist()
    actualIndices = np.where(test_labels[j] > 0)[0]
    actualLabels = names[actualIndices]

    predictedVector = np.zeros_like(test_labels[j])
    predictedIndices = np.where(predicted[j] > 0.5)[0] # prob > 0.5
    predictedLabels = names[predictedIndices]
    predictedProbabilities = predicted[j][predictedIndices]
    predictedVector[predictedIndices] = 1

    predResult['sampleIndex'] = j
    predResult['sentence'] = sentence
    predResult['actualLabels'] = actualLabels.tolist()
    predResult['predictedLabels'] = predictedLabels.tolist()
    predResult['predictedProbabilities'] = predicted[j].tolist()
    predResults.append(predResult)

    if not (np.array_equal(test_labels[j],predictedVector)):
        print ('index MISMATCH Actual/Predicted:',j, actualLabels, '/', predictedLabels, 'Probabilities:', predictedProbabilities)
    totalError = totalError + np.abs(test_labels[j] - predictedVector)

result['predictions'] = predResults
print ('TotalError/Sum:', totalError,np.sum(totalError))
result['totalError'] = totalError.tolist()
result['totalErrorSum'] = int(np.sum(totalError))

f = open ('results/' + attention + '-' + objective + '-' + mask + '.json','w')
out = json.dumps(result, ensure_ascii=True)
f.write(out)
f.close()

np.set_printoptions()

