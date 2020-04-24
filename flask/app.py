from flask import Flask, request, jsonify
import pickle
import os
import string
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from apptokenizer import tokenizer
from pathlib import Path
import model

app = Flask(__name__)


# prediction happens here
def vectorize_user_input(user_input_text):
    pickle_file = tf_idf_file

    print(os.getcwd())
    print('PICKLE')
    print(pickle_file)

    if not os.path.exists(pickle_file):
        return 'Pickle file not found'

    with open(pickle_file, "rb") as file:
        print('---get vectorizer from pickle')
        vectorizer = pickle.load(file)
        user_input_text = user_input_text.lower()
        user_input_text = ''.join(c for c in user_input_text if c not in string.punctuation)  # remove punctuations
        user_input_text = ''.join(c for c in user_input_text if c not in '0123456789')  # remove digits
        user_input_text = ' '.join(user_input_text.split())  # trim extra spaces
        user_input_text = user_input_text.split()
        sparse_texts = vectorizer.transform(user_input_text)

        print(' SPARSE TEXT ')
        print(sparse_texts[0])
        return sparse_texts[0]


@app.route('/api/get_text_prediction', methods=['POST'])
def get_text_prediction():
    """
    predicts requested text whether it is ham or spam
    :return: json
    """

    json = request.get_json()
    print(json)

    if len(json['text']) == 0:
        return jsonify({'error': 'invalid input'})

    print('---user_data before processing :: ', json['text'])
    user_input_text = vectorize_user_input(json['text'])
    print('---user_data after vectorized user_input_text.shape :: ', user_input_text.shape)
    print('---user_data after vectorized user_input_text.todense() :: ', user_input_text.todense())

    # load the model
    graph = tf.Graph()
    with graph.as_default():
        sess = tf.Session()
        with sess.as_default():
            # Load the saved meta graph and restore variables
            meta_graph = MODEL_NAME + '_model.ckpt'
            saver = tf.train.import_meta_graph("{}.meta".format(Path(out_dir / meta_graph)))
            sess_store = MODEL_NAME + '_model.ckpt'
            saver.restore(sess, str(Path(out_dir / sess_store)))

            # Get the placeholders from the graph by name
            input_ = graph.get_operation_by_name(input_node_name).outputs[0]
            output_ = graph.get_operation_by_name(output_node_name).outputs[0]
            prediction_ = graph.get_operation_by_name(prediction_node_name).outputs[0]

            # Make the prediction
            # value = sess.run(prediction_, feed_dict={input_: user_input_text.todense()})
            value = sess.run(output_, feed_dict={input_: user_input_text.todense()})
            print('---TTT :: value :: ', value)
            print('---TTT :: len(value) :: ', len(value))
            print('---TTT :: value[0] :: ', value[0], value)
            print(output_[0])

    return jsonify({
        'you sent this': 'thank you!',
        '---user_data before processing :: ': str(json['text']),
        '---user_data after vectorized user_input_text.shape :: ': str(user_input_text.shape),
        '---TTT :: value :: ': str(value),
        '---TTT :: len(value) :: ': str(len(value)),
        '---TTT :: value[0] :: ': str(value[0]),
    })


@app.route("/")
def index():
    """
    this is a root dir of my server
    :return: str
    """
    return "Test!!!!"


@app.route('/users/<user>')
def hello_user(user):
    """
    this serves as a demo purpose
    :param user:
    :return: str
    """
    return "Hello %s!" % user


if __name__ == '__main__':

    out_dir = Path('../out')
    tf_idf_file = Path('../tfidf.pickle')
    MODEL_NAME = 'spam_ham_text_classifier'
    model_ckpt = '_model.ckpt'

    input_node_name = 'MODEL_INPUTT'
    output_node_name = 'MODEL_OUTPUTT'
    prediction_node_name = 'MODEL_PREDICTION'

    model.create_model()
    app.run(host='192.168.1.5', port=18080)
