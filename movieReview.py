from tensorflow import keras
import numpy as np

__name__ = "__main__"
data = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=88000)

word_index = data.get_word_index()

word_index = {k: (v+3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

# Limit of 250 words per review, adding padding for < 250
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"],
                                                        padding="post", maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"],
                                                       padding="post", maxlen=250)


def decode_review(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])


def build_model():
    # model
    model = keras.Sequential()
    model.add(keras.layers.Embedding(88000, 16)),
    model.add(keras.layers.GlobalAveragePooling1D()),  # Coverts to 1 dim
    model.add(keras.layers.Dense(16, activation="relu")),
    model.add(keras.layers.Dense(1, activation="sigmoid"))

    model.summary()

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    x_val = train_data[:10000]
    x_train = train_data[10000:]

    y_val = train_labels[:10000]
    y_train = train_labels[10000:]

    fitModel = model.fit(x_train, y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val), verbose=1)

    results = model.evaluate(test_data, test_labels)

    print(results)

    model.save("model.h5")
    return model


def review_encode(s):
    encoded = [1]
    for word in s:
        if word.lower() in word_index:
            encoded.append(word_index[word.lower()])
        else:
            encoded.append(2)  # unknown word
    return encoded


def make_prediction():
    try:
        f = open("model.h5")
        model = keras.models.load_model("model.h5")
    except IOError:
        model = build_model()
    finally:
        f.close()

    with open("test.txt", encoding="utf-8") as f:
        for line in f.readlines():
            nline = line.replace(",", "").replace(".", "").replace("(", "").replace(")", "").replace(":", "") \
                .replace("\"", "").strip().split(" ")
            encode = review_encode(nline)
            encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index["<PAD>"], padding="post",
                                                                maxlen=250)
            predict = model.predict(encode)
            print(line)
            print(encode)
            percentage = np.round(predict[0], 2)
            if 0.00 < percentage <= 0.25:
                print("This review is very bad!")
            elif 0.25 < percentage <= 0.50:
                print("This review is bad!")
            elif 0.50 < percentage <= 0.75:
                print("This review is good!")
            elif 0.75 < percentage <= 1.00:
                print("This review is excellent!")
            else:
                print("Unknown review!")
            print(percentage)


if __name__ == "__main__":
    make_prediction()

