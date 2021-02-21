import keras
import numpy
import pandas
import time

Data = pandas.read_csv("fra.txt", sep = "\t", names = ["EN","FR"], usecols=[0,1], nrows=10000)

Data['EN'] = Data['EN'].apply( lambda x : '\t{}\n'.format(x) )  ## Ajout des jetons de début et  de fin
Data['FR'] = Data['FR'].apply( lambda x : '\t{}\n'.format(x) )  ## Ajout des jetons de début et  de fin

Tokenizer = keras.preprocessing.text.Tokenizer(lower=True, char_level=True)
Tokenizer.fit_on_texts(list(Data['EN']) + list(Data['FR']))

Data['EN_enc'] = Tokenizer.texts_to_sequences(Data['EN'])
Data['FR_enc'] = Tokenizer.texts_to_sequences(Data['FR'])


num_tokens = len(Tokenizer.word_index) + 1 # +1 car l'indice des jeton commence à 1
embedding_units = 32
lstm_units = 1024

def createEncoder():
    inputs = keras.layers.Input(shape=(None,))
    embed_layer = keras.layers.Embedding(input_dim=num_tokens, output_dim=embedding_units)
    embed = embed_layer(inputs)

    lstm_layer = keras.layers.LSTM(units=lstm_units, return_sequences=False, return_state=True)
    _, h, c = lstm_layer(embed)

    return keras.models.Model(inputs, [h, c])

def createDecoder():
    inputs = keras.layers.Input(shape=(None,))
    embed_layer = keras.layers.Embedding(input_dim=num_tokens, output_dim=embedding_units)
    embed = embed_layer(inputs)
    input_h = keras.layers.Input(shape=(lstm_units,))
    input_c = keras.layers.Input(shape=(lstm_units,))
        
    lstm_layer = keras.layers.LSTM(units=lstm_units, return_sequences = True, return_state=True)
    lstm_out, h , c = lstm_layer(embed, initial_state=[input_h, input_c])

    dense_layer = keras.layers.Dense(num_tokens, activation='softmax')
    out = dense_layer(lstm_out)

    return keras.models.Model( [inputs, input_h, input_c], [out, h, c] )

def createTrainingModel(Encoder, Decoder):
    out, h, c = Decoder( (Decoder.input[0], Encoder.output[0], Encoder.output[1]) )  # Note : Decoder.input[0] = inputs, Encoder.output[0] = h, Encoder.output[1] = c

    model = keras.models.Model([Encoder.input, Decoder.input[0]], out)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

    return model


Encoder = createEncoder()
Decoder = createDecoder()
TrainingModel = createTrainingModel(Encoder, Decoder)
        
## Trainning

input_encoder = keras.preprocessing.sequence.pad_sequences(Data["EN_enc"], maxlen=None, dtype='int32', padding='post', value=Tokenizer.word_index['\n'])
input_decoder = keras.preprocessing.sequence.pad_sequences(Data["FR_enc"], maxlen=None, dtype='int32', padding='post', value=Tokenizer.word_index['\n'])
    
target = numpy.roll(input_decoder, shift=-1)  ## Décalage vers la gauche
target[:,-1] = Tokenizer.word_index['\n']     ## Correction sur le dernier jeton

TrainingModel.fit([input_encoder, input_decoder], target, epochs=1000, verbose = 2)

## Prédiction

MAX_LENGTH = input_encoder.shape[1] ## dimension timestep 

def Translate(phrase):
    ## Ajout des jetons de début et de fin
    phrase = '\t' + phrase + '\n'   

    ## Encodage des jetons
    encoder_input_data = Tokenizer.texts_to_sequences([phrase])  # format [batch, timestep] avec batch=1

    ## ajustement de la dimension timestep
    encoder_input_data = keras.preprocessing.sequence.pad_sequences(encoder_input_data, maxlen=MAX_LENGTH,
                                dtype='int32', padding='post', truncating='post', value=tokenizer_EN.word_index['\n'])  

    ## Récupération de la sortie de l'encodeur
    h, c = Encoder.predict(encoder_input_data)  # format [batch, feature] avec batch=1 et feature=1024

    char = Tokenizer.word_index['\t'] ## Jeton <START>
    result = list()
    result.append(char)
    while char != tokenizer_FR.word_index['\n'] and len(result) <1000: ## Tant que char != <END> ou result pas trop grand
        input_dec = np.array([[char]])  ## input_dec : format [batch, timestep] avec batch=1, timestep=1,
        out ,h ,c = Decoder.predict([input_dec ,h ,c])  ## out : format [batch, timestep, feature] avec batch=1, timestep=1, feature=num_tokens
        char = np.argmax(out[0,-1,:])
        result.append(char)

    return ''.join([Tokenizer.index_word[x] for x in result])  ## Conversion des jetons encodés en caractères
    

DT = Data.loc[0:N,['EN']].sample(100).copy()
DT['pred'] = DT['EN'].apply( lambda x : Translate(x)  )

