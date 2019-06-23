import sys

import tensorflow as tf

from config import Config
from utils import check_restore_params
from create_model import create_model
from data import generate_batches
from data import load_dataset 
from data import query2batch
from seq2seq_model import Seq2Seq_Model

config = Config()


def inference(): 
    with tf.Session() as sess:
        model = create_model(config) 
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(config.save_path)
        if ckpt and tf.train.get_checkpoint_state(ckpt.model_checkpoint_path):
            print("start to load parameters")
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('finish loading parameters')
        else:
            #raise ValueError('No model saved at this path:{}'.format(config.save_path))
            print("start to load parameters")
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('finish loading parameters')

        sys.stdout.write('Please type your query:')
        sys.stdout.flush()
        
        query = sys.stdin.readline()
        while query:
            if not query:
                pass
            else: 
                data_batch = query2batch(query, config.word2idx) 
            
            infered_ids = model.step(sess, data_batch, forward_only=True, mode='inference')
            response = ''
            for idx in infered_ids[0]:
                response += (config.idx2word[idx] + ' ')
            print('The generated response is: {}'.format(response))
            sys.stdout.flush()
            query = sys.stdin.readline()
        


if __name__ == '__main__':
    tf.reset_default_graph()
    inference()
