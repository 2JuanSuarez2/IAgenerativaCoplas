import tensorflow as tf
import numpy as np

# Conjunto de datos de entrenamiento (coplas cojas)
coplas = [
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

    "Me fui al limbo",
    "a hablar con los muertos",
    "pero como no me entendían",
    "me fui a casa a comer.",

    "Me fui al mundo de los sueños",
    "a buscar a mi amor",
    "pero como no lo encontré",
    "me fui a casa a llorar.",

    "Me fui a casa a dormir",
    "y soñé que volaba",
    "pero cuando desperté",
    "me di cuenta de que era el ventilador.",

    "Me fui al baño a hacer pis",
    "y me resbalé",
    "y me caí en la taza",
    "y me caí de rodillas.",

    "Me fui a la cocina a comer",
    "y me quemé la lengua",
    "y me quedé sin habla",
    "y tuve que comer con los ojos.",

    "Me fui al cine a ver una película",
    "y me quedé dormido",
    "y cuando desperté",
    "ya había empezado otra.",

    "Me fui de vacaciones",
    "y me olvidé el pasaporte",
    "y me tuve que quedar",
    "en mi propia casa.",

    "Me fui de compras",
    "y me gasté todo el dinero",
    "y me quedé sin nada",
    "para comer ni para vestirme.",

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
    "todavía me quedaba mucho por vivir.",
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

# This
# Capa de Dropout: Agregar capas de dropout para evitar el sobreajuste.
# El sobreajuste podría estar causando la repetición excesiva de palabras.

# Construir y entrenar el modelo (LSTM con Dropout)
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(total_words, 100, input_length=max_sequence_length - 1))
model.add(tf.keras.layers.LSTM(150, dropout=0.2, recurrent_dropout=0.2))
model.add(tf.keras.layers.Dense(total_words, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=100
          , verbose=1)


# Función para generar coplas
def generate_copla(seed_text, model, max_sequence_length):
    generated_text = seed_text
    for _ in range(max_sequence_length):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = tf.keras.preprocessing.sequence.pad_sequences([token_list], maxlen=max_sequence_length - 1,
                                                                   padding='pre')

        predicted_probs = model.predict(token_list, verbose=0)
        # This
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