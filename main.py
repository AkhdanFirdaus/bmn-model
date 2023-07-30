import numpy as np
from flask import Flask, request, jsonify
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


class Preprocessing():
    def __init__(self, stemmer, stopword, tokenizer, max_len=128):
        self.stemmer = stemmer
        self.stopword = stopword
        self.tokenizer = tokenizer
        self.max_len = max_len

    def casefolding(self, val):
        return str(val).lower()

    def stemming(self, val):
        return self.stemmer.stem(str(val))

    def stopwordremove(self, val):
        return self.stopword.remove(str(val))

    def preprocessing(self, sentences):
        input_ids, attention_mask = [], []
        for sentence in sentences:
            input = self.casefolding(sentence)
            input = self.stemming(input)
            input = self.stopwordremove(input)
            tokenized = tokenizer.encode_plus(
                input,
                add_special_tokens=True,
                max_length=self.max_len,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_token_type_ids=False,
                return_tensors='tf'
            )

            input_ids.append(tokenized['input_ids'])
            attention_mask.append(tokenized['attention_mask'])

        return {
            'input_ids': tf.convert_to_tensor(np.asarray(input_ids).squeeze(), dtype=tf.int32),
            'attention_mask': tf.convert_to_tensor(np.asarray(attention_mask).squeeze(), dtype=tf.int32)
        }


class Process():
    def __init__(self, model):
        self.model = model
        self.threshold = 0.5

    def rounded_predictions(self, inputs):
        predictions = self.model.predict(inputs)
        return np.where(predictions > self.threshold, 1, 0)

    def measure_severity(self, inputs, labels):
        return {}

    def predict(self, inputs, labels):
        return {
            'data': {
                'severity': NULL,
                'results': [],
            }
        }


app = Flask(__name__)

stemmer = StemmerFactory().create_stemmer()
stopword = StopWordRemoverFactory().create_stop_word_remover()
tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p1')
preprocess = Preprocessing(stemmer, stopword, tokenizer, 128)

loaded_model = tf.keras.models.load_model(
    './model/klasifikasi2.h5',
    custom_objects={'TFBertModel': TFBertModel},
    compile=False
)
process = Process(loaded_model)


@app.route('/hello', methods=['GET'])
def hello():
    return jsonify({'message': 'Hello World!'})


@app.route('/predict', methods=['POST'])
def predict():
    # labels = request.form['labels']
    incoming_request = request.get_json()
    inputs = incoming_request.get('inputs')
    tokenized = preprocess.preprocessing(inputs)
    predictions = process.rounded_predictions(tokenized)
    return jsonify({'data': predictions})


if __name__ == '__main__':
    app.run(debug=True)
