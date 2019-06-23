import tensorflow as tf
import os



def create_rnn_cell(rnn_type, num_layers, rnn_size, output_keep_prob):
    return tf.contrib.rnn.MultiRNNCell([create_single_rnn_layer(rnn_type, rnn_size, output_keep_prob) for _ in range(num_layers)])


def create_single_rnn_layer(rnn_type, rnn_size, output_keep_prob):
    if rnn_type == 'LSTM':
        cell = tf.contrib.rnn.LSTMCell(rnn_size)
        drop_cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=output_keep_prob)
        return drop_cell
    else:
        raise NotImplementedError

def create_attention_mechanism(attention_type, num_units, memory):
    if attention_type == 'Bahdanau':
        return tf.contrib.seq2seq.BahdanauAttention(num_units=num_units, memory=memory)
    else:
        raise NotImplementedError

def check_restore_params(path, sess, saver):
    ckpt = tf.train.get_checkpoint_state(os.path.dirname(path))
    if ckpt and ckpt.model_checkpoint_path:
        print "loading from existing model"
        saver.retore(sess, ckpt.model_checkpoint_path)
    else:
        print("initilizing new seq2seq-chatbot")
        
