from seq2seq_model import Seq2Seq_Model

def create_model(config):
    model = Seq2Seq_Model(vocab_size = config.vocab_size,
                          output_keep_prob = config.output_keep_prob,
                          encoder_rnn_type = config.encoder_rnn_type,
                          encoder_num_layers = config.encoder_num_layers,
                          encoder_rnn_size = config.encoder_rnn_size,
                          encoder_embedding_size = config.encoder_embedding_size,
                          decoder_rnn_type = config.decoder_rnn_type,
                          decoder_num_layers = config.decoder_num_layers,
                          decoder_rnn_size = config.decoder_rnn_size,
                          decoder_embedding_size = config.decoder_embedding_size,
                          use_attention = config.use_attention,
                          attention_type = config.attention_type,
                          opt = config.opt,
                          clip_norm = config.clip_norm,
                          learning_rate = config.learning_rate,
                          word2idx = config.word2idx)
    return model
