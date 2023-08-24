import tensorflow as tf
import numpy as np

# Conjunto de datos de entrenamiento (coplas cojas)
coplas = [
    "ayer pasé por tu casa",
    "y me tiraste un hueso de pollo",
    "como me asusté bastante",
    "se me puso la piel de gallina.",
    "ayer pasé por tu casa",
    "y me tiraste un telescopio,",
    "si no es porque yo me agacho",
    "me haces ver estrellitas.",
    "ayer pasé por tu casa",
    "me tiraste un televisor",
    "y yo hice sam-sung",
    "y lo esquive."
]

# Preprocesamiento de datos
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(coplas)
total_words = len(tokenizer.word_index) + 1

input_sequences = []
for line in coplas:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i + 1]
        input_sequences.append(n_gram_sequence)

max_sequence_length = max(len(seq) for seq in input_sequences)
input_sequences = tf.keras.preprocessing.sequence.pad_sequences(input_sequences, maxlen=max_sequence_length)

# Separar datos de entrada y etiquetas
input_sequences = np.array(input_sequences)
X, y = input_sequences[:, :-1], input_sequences[:, -1]
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

# Capa de Dropout: Agregar capas de dropout para evitar el sobreajuste.
# El sobreajuste podría estar causando la repetición excesiva de palabras.

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
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = tf.keras.preprocessing.sequence.pad_sequences([token_list], maxlen=max_sequence_length - 1,
                                                                   padding='pre')

        predicted_probs = model.predict(token_list, verbose=0)

        # Utilizar muestreo aleatorio ponderado
        predicted_index = np.random.choice(len(predicted_probs[0]), p=predicted_probs[0])

        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                output_word = word
                break

        # Evitar repeticiones inmediatas
        if output_word == seed_text.split()[-1]:
            continue

        generated_text += " " + output_word
        seed_text = generated_text
    return generated_text


# Semillas para generar coplas
seed_texts = ["ayer pasé", "y me tiraste", "como me asusté", "ayer pasé por tu casa", "y me tiraste un telescopio,",
              "si no es porque yo me agacho", "ayer pasé por tu casa", "me tiraste un televisor", "y yo hice sam-sung"]

# Generar y mostrar coplas completas
for seed_text in seed_texts:
    generated_copla = generate_copla(seed_text, model, max_sequence_length)
    print(f"Semilla: {seed_text}\nGenerada: {generated_copla}\n")