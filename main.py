import tensorflow as tf
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist

# Descargar los datos necesarios para NLTK si aún no lo has hecho
nltk.download('punkt')

# Conjunto de datos de entrenamiento (coplas cojas)
coplas = [
    "ayer pasé por tu casa",
    "y me tiraste un hueso de pollo",
    "como me asusté bastante",
    "se me puso la piel de gallina",
    "ayer pasé por tu casa",
    "y me tiraste un telescopio,",
    "si no es porque yo me agacho",
    "me haces ver estrellitas",
    "ayer pasé por tu casa",
    "me tiraste un televisor",
    "y yo hice sam-sung",
    "y lo esquive",
    "ayer pase por tu casa",
    "me tiraste una estufa",
    "si no me agacho paila"
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
def generate_copla(seed_text, model, max_sequence_length):
    generated_text = seed_text
    for _ in range(max_sequence_length):
        token_list = word_tokenize(seed_text)
        token_indices = [word_to_index[word] for word in token_list]
        token_indices = tf.keras.preprocessing.sequence.pad_sequences([token_indices], maxlen=max_sequence_length - 1,
                                                                   padding='pre')

        predicted_probs = model.predict(token_indices, verbose=0)
        predicted_index = np.random.choice(len(predicted_probs[0]), p=predicted_probs[0])

        output_word = ""
        for word, index in word_to_index.items():
            if index == predicted_index:
                output_word = word
                break

        if output_word == seed_text.split()[-1]:
            continue

        generated_text += " " + output_word
        seed_text = generated_text
    return generated_text

# Semillas para generar coplas
seed_texts = ["ayer pasé", "y me tiraste", "como me asusté", "ayer pasé por tu casa", "y me tiraste un telescopio,",
              "si no es porque yo me agacho", "ayer pasé por tu casa", "me tiraste un televisor", "y yo hice sam-sung",
              "ayer pase","me tiraste","si no me agacho"]

# Generar y mostrar coplas completas
for seed_text in seed_texts:
    generated_copla = generate_copla(seed_text, model, max_sequence_length)
    print(f"Semilla: {seed_text}\nGenerada: {generated_copla}\n")