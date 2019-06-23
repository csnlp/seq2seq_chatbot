from data import load_dataset

class Config(object):
    ## Data
    filename = './dataset/dataset_cornell_p2.pkl'
    vocab_size = 5000
    word2idx, idx2word, dialogs = load_dataset(filename)
    
    ## drop out
    output_keep_prob = 1
    
    ## Encoder
    encoder_rnn_type = 'LSTM'
    encoder_num_layers = 3
    encoder_rnn_size = 1024
    encoder_embedding_size = 1024

    # Decoder
    decoder_rnn_type = 'LSTM'
    decoder_num_layers = 3
    decoder_rnn_size = 1024
    decoder_embedding_size = 1024

    ## Attention
    use_attention = True
    attention_type = 'Bahdanau'

    ## Optimizer
    opt = 'adam'
    clip_norm = 0.5
    learning_rate = 0.00001

    ## training
    epoches = 30
    batch_size = 128

    ## save model 
    save_epoch = 1
    save_path = 'models/'
    save_name = 'seq2seq_chatbot.ckpt'
