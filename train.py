import tensorflow as tf

from config import Config
from utils import check_restore_params
from create_model import create_model
from data import generate_batches
from data import load_dataset 
from seq2seq_model import Seq2Seq_Model

config = Config()

def train():
    with tf.Session() as sess:
        model = create_model(config)
        sess.run(tf.global_variables_initializer())

        ## BE SURE THAT "[saver = tf.train.Saver()]" is after "[model = create_model(config)]"
        saver = tf.train.Saver()
        
        step = 0
        print('start training')
        
        for no_epoch in range(1, config.epoches):
            batches = generate_batches(config.dialogs, config.batch_size) 
            for no_batch in range(1, len(batches) + 1):
                _, loss = model.step(sess, batches[no_batch - 1], forward_only=False, mode='train')
                if step % 20 == 0:
                    print('step{}'.format(step) + 'batch loss:{}'.format(loss))

                step = step + 1

            if no_epoch % config.save_epoch == 0:
                saver.save(sess, config.save_path + config.save_name, global_step=step)
                print('model saved at step ={}'.format(step))

        print('finish training')


if __name__ == '__main__':
    train()
