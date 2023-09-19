import tensorflow as tf
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist

# Descargar los datos necesarios para NLTK
# nltk.download('punkt')

# Conjunto de datos de entrenamiento (coplas cojas)
coplas = [
    "ayer pasé por tu casa",
    "tiraste un hueso de pollo",
    "me puso piel de gallina",

    "ayer pasé por tu casa",
    "y me tiro un telescopio,",
    "y si no me agacho",
    "me haces ver muchas estrellitas",

    "ayer pasé por tu casa",
    "usted tiro un gran televisor",
    "y yo hice sam-sung",
    "y lo esquive",

    "ayer pase por tu casa",
    "me tiraste una pequeña estufa",
    "si no me agacho paila",

    "Me comí un león",
    "y me salió un diente de oro",
    "y me lo puse en la boca",
    "y me quedé sin hambre.",

    "Me fui a nadar al mar",
    "y me encontré un tiburón",
    "le dije: '¿Qué haces aquí?'",
    "y me dijo: '¡Nadando!'",

    "Me fui a una discoteca",
    "y me puse a bailar",
    "pero como no sabía bailar",
    "me puse a llorar.",

    "Me fui a un restaurante",
    "y pedí una tortilla",
    "pero cuando me la trajeron",
    "no tenía sal.",

    "Me fui a la escuela",
    "y me puse a estudiar",
    "pero como no sabía estudiar",
    "me puse a jugar.",

    "Me fui a trabajar",
    "y me puse a descansar",
    "pero como no tenía trabajo",
    "me puse a llorar.",

    "Me fui a casa",
    "y me puse a dormir",
    "pero como no tenía cama",
    "me puse a llorar.",

    "Me fui a la cama",
    "y me puse a soñar",
    "pero como no sabía soñar",
    "me puse a llorar.",

    "Me enamoré de un pez",
    "que vivía en el mar",
    "para ir a verlo",
    "me comí un calamar.",

    "Me fui a la montaña",
    "a buscar un diamante",
    "pero cuando llegué",
    "ya lo había encontrado un vidente.",

    "Me fui a la luna",
    "a buscar un astronauta",
    "pero cuando llegué",
    "ya estaba de vuelta en la cama.",

    "Me fui a un partido de fútbol",
    "y me puse a cantar",
    "pero como no sabía cantar",
    "me expulsaron del estadio.",

    "Me fui a la playa",
    "a buscar un cangrejo",
    "pero cuando llegué",
    "ya estaba en un restaurante.",

    "Me fui a un circo",
    "a ver a un payaso",
    "pero como no era gracioso",
    "me fui a casa a dormir.",

    "Me fui a un concierto",
    "a escuchar a un cantante",
    "pero como no sabía cantar",
    "me puse a bailar.",

    "Me fui a un funeral",
    "a despedir a un amigo",
    "pero como no sabía llorar",
    "me puse a reír.",

    "Me fui al cielo",
    "a hablar con Dios",
    "pero como no me dejó pasar",
    "me quedé en la tierra.",

    "Me fui al infierno",
    "a hablar con el diablo",
    "pero como no tenía nada que hacer",
    "me fui a casa a dormir.",

    "Me fui a casa a dormir",
    "y soñé que volaba",
    "pero cuando desperté",
    "me di cuenta de que era el ventilador.",

    "Me fui a la cocina a comer",
    "y me quemé la lengua",
    "y me quedé sin habla",
    "y tuve que comer con los ojos.",

    "Me fui a la cama",
    "y me quedé sin sueño",
    "y me tuve que levantar",
    "a ver si la noche pasaba.",

    "Me fui a vivir solo",
    "y me aburrió",
    "y me fui a vivir con mis padres",
    "que me hacen la comida.",

    "Me fui a morir",
    "y me volví a levantar",
    "y me di cuenta de que",
    "todavía me quedaba mucho por vivir."
]

# Preprocesamiento de datos con NLTK
all_words = []
for line in coplas:
    token_list = word_tokenize(line)  # Tokenizar con NLTK
    all_words.extend(token_list)

# Crear un vocabulario y convertir palabras a números
word_freq = FreqDist(all_words)
word_to_index = {word: idx for idx, (word, _) in enumerate(word_freq.items())}
total_words = len(word_to_index)

# Crear secuencias de entrada y etiquetas
input_sequences = []
for line in coplas:
    token_list = word_tokenize(line)
    token_indices = [word_to_index[word] for word in token_list]
    for i in range(1, len(token_indices)):
        n_gram_sequence = token_indices[:i + 1]
        input_sequences.append(n_gram_sequence)

# Pad secuencias para que tengan la misma longitud
max_sequence_length = max(len(seq) for seq in input_sequences)
input_sequences = tf.keras.preprocessing.sequence.pad_sequences(input_sequences, maxlen=max_sequence_length)

# Separar datos de entrada y etiquetas
X, y = input_sequences[:, :-1], input_sequences[:, -1]

# Convertir etiquetas a one-hot encoding
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

# Construir y entrenar el modelo (LSTM con Dropout)
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(total_words, 100, input_length=max_sequence_length - 1))
model.add(tf.keras.layers.LSTM(150, dropout=0.2, recurrent_dropout=0.2))
model.add(tf.keras.layers.Dense(total_words, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=100, verbose=1)

# Función para generar coplas
def generate_copla(seed_text, model, max_sequence_length, max_length=5):
    generated_text = seed_text
    for _ in range(max_length):
        token_list = word_tokenize(seed_text)
        token_indices = [word_to_index[word] for word in token_list]
        token_indices = tf.keras.preprocessing.sequence.pad_sequences([token_indices], maxlen=max_sequence_length - 1,
                                                                   padding='pre')

        predicted_probs = model.predict(token_indices, verbose=0)
        predicted_probs[0][word_to_index[","]] = 0.0
        
        # Normalizar las probabilidades
        predicted_probs = predicted_probs / predicted_probs.sum()

        predicted_index = np.random.choice(len(predicted_probs[0]), p=predicted_probs[0])

        output_word = ""
        for word, index in word_to_index.items():
            if index == predicted_index:
                output_word = word
                break

        if output_word == seed_text.split()[-1] or output_word == ",":
            continue

        generated_text += " " + output_word
        seed_text = generated_text
        
        # Verifica si se ha alcanzado la longitud máxima
        if len(generated_text.split()) >= max_length:
            break
    
    return generated_text


# Semillas para generar coplas
seed_texts = ["ayer pasé", "tiraste un hueso", "me puso", "ayer pasé", "y me tiro un",
              "y si no","me haces", "ayer pasé", "usted tiro", "y yo hice",
              "ayer pase","me tiraste una","si no me agacho","me fui a una","y me puse a","pero como no",
              "y pedí una","pero cuando me","no tenía","me comí un","y me salió un","y me lo puse en",
              "y me quedé","me fui a nadar","y me encontré","le dije:","y me dijo:"]

# Generar y mostrar coplas completas
for seed_text in seed_texts:
    generated_copla = generate_copla(seed_text, model, max_sequence_length)
    print(f"Semilla: {seed_text}\nGenerada: {generated_copla}\n")