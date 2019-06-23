import tensorflow as tf

from utils import create_rnn_cell
from utils import create_attention_mechanism



class Seq2Seq_Model(object):
    def __init__(self, 
                 vocab_size, 
                 output_keep_prob,
                 encoder_rnn_type, 
                 encoder_num_layers, 
                 encoder_rnn_size, 
                 encoder_embedding_size, 
                 decoder_rnn_type, 
                 decoder_num_layers, 
                 decoder_rnn_size, 
                 decoder_embedding_size, 
                 use_attention, 
                 attention_type, 
                 opt, 
                 clip_norm,
                 learning_rate, 
                 word2idx):

        self.vocab_size = vocab_size

        self.output_keep_prob = output_keep_prob

        self.encoder_rnn_type = encoder_rnn_type
        self.encoder_num_layers = encoder_num_layers
        self.encoder_rnn_size = encoder_rnn_size
        self.encoder_embedding_size = encoder_embedding_size

        self.decoder_rnn_type = decoder_rnn_type
        self.decoder_num_layers = decoder_num_layers
        self.decoder_rnn_size = decoder_rnn_size
        self.decoder_embedding_size = decoder_embedding_size
        
        self.use_attention = use_attention
        self.attention_type = attention_type
        
        self.opt = opt
        self.clip_norm = clip_norm
        self.learning_rate = learning_rate


        self.word2idx = word2idx

        self.build_graph()

    def build_graph(self):
        print ('#' * 48 + '\n \n Begin to build Seq2Seq model \n \n' + '#' * 48)
        
        ## add placeholders
        batch_size = tf.placeholder(tf.int32, [])
        self.batch_size = batch_size
        # self.encoder_sources = tf.placeholder(tf.int32, [self.batch_size, None])
        encoder_sources = tf.placeholder(tf.int32, [None, None])
        self.encoder_sources = encoder_sources
        decoder_targets = tf.placeholder(tf.int32, [None, None])
        self.decoder_targets = decoder_targets

        
        sources_length_list = tf.placeholder(tf.int32, [None])
        self.sources_length_list = sources_length_list
    
        targets_length_list = tf.placeholder(tf.int32, [None])
        self.targets_length_list = targets_length_list
        

        # build encoder
        with tf.variable_scope('Encoder'):
            # since our seq2seq in chatbot locates in monolingual corpus, 
            # source and target sequences share the embeddings matrix
            embedding = tf.get_variable(name = 'embedding_matrix', 
                            shape = [self.vocab_size, self.encoder_embedding_size])
            encoder_cell = create_rnn_cell(self.encoder_rnn_type, self.encoder_num_layers, 
                                    self.encoder_rnn_size, self.output_keep_prob)
            ## encoder_embedding_output: batch_size * source_length * embedding_size 
            embedded_encoder_input = tf.nn.embedding_lookup(embedding, self.encoder_sources)
            # encoder_outputs: [batch_size, source_length, embedding_size], encoder_state: [batch_size, embedding_size]
            encoder_outputs, encoder_state = tf.nn.dynamic_rnn(cell=encoder_cell, 
                                        inputs=embedded_encoder_input, dtype=tf.float32)
            

        # build decoder
        with tf.variable_scope('Decoder'):
            attention_mechanism = create_attention_mechanism(self.attention_type, 
                                    self.encoder_rnn_size, encoder_outputs)
            # decoder cell
            decoder_cell = create_rnn_cell(self.decoder_rnn_type, self.decoder_num_layers, 
                                self.decoder_rnn_size, self.output_keep_prob)
            attention_decoder_cell = tf.contrib.seq2seq.AttentionWrapper(cell=decoder_cell, 
                                        attention_mechanism=attention_mechanism)


            ## self.mode == 'train':
            # concate
            decoder_input = tf.concat([tf.fill([self.batch_size, 1], self.word2idx['<go>']), self.decoder_targets], 1)
            ## delete the end symbol such as <EOS>
            decoder_input = tf.strided_slice(input_=decoder_input, begin=[0, 0], 
                            end=[self.batch_size, -1], strides=[1,1])
            embedded_decoder_input = tf.nn.embedding_lookup(embedding, decoder_input)
            
            training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=embedded_decoder_input, 
                                sequence_length=self.targets_length_list)
            projection_layer = tf.layers.Dense(units=self.vocab_size)
            
            ## judge initial state of the decoder
            if self.use_attention:
                decoder_initial_state = attention_decoder_cell.zero_state(batch_size=self.batch_size, 
                                            dtype=tf.float32).clone(cell_state=encoder_state)
            else:
                decoder_initial_state = encoder_state

            training_decoder = tf.contrib.seq2seq.BasicDecoder(cell=attention_decoder_cell, 
                        helper=training_helper, initial_state=decoder_initial_state, output_layer=projection_layer)                

            # decoder_outputs (rnn_output: [batch_size, targets_length, vocab_size], 
            # sample_id: [batch_size] this is final output of decoder)
            decoder_outputs, decoder_state, decoder_sequence_length = \
                        tf.contrib.seq2seq.dynamic_decode(training_decoder)
            decoder_logits = tf.identity(decoder_outputs.rnn_output)

            # mask:[batch_size, max_decoder_targets_length], mask is used to calculate the loss function: 
            # for pedding position, the weight of this position loss is zero
            mask = tf.sequence_mask(self.targets_length_list)
            mask = tf.cast(mask, tf.float32)
            # decoder_logits: [batch_size, target_sequence_length, vocal_size], 
            # targets: [batch_size, target_sequence_length]
            self.loss = tf.contrib.seq2seq.sequence_loss(logits=decoder_logits, 
                    targets=self.decoder_targets, weights=mask)

            optimizer = tf.train.AdamOptimizer(self.learning_rate)

            params = tf.trainable_variables()
            gradients = tf.gradients(self.loss, params)
            clip_gradients, _ = tf.clip_by_global_norm(gradients, self.clip_norm)
            self.train_op = optimizer.apply_gradients(zip(clip_gradients, params))
                
            ## **** inference stage **** ##
            # start_tokens = tf.constant(value=self.word2idx['<go>'], shape=[self.batch_size])
            start_tokens = tf.cast(tf.ones([self.batch_size]) * self.word2idx['<go>'], tf.int32)
            end_token = self.word2idx['<eos>']
            greedy_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding=embedding, 
                                start_tokens=start_tokens, end_token=end_token)
 
            inference_decoder = tf.contrib.seq2seq.BasicDecoder(cell=attention_decoder_cell, 
                            helper=greedy_helper, initial_state=decoder_initial_state, output_layer=projection_layer)                
            # inference_outputs: (rnn_output : [batch_size, decoder_targets_length, vocab_size], 
            #                     sample_id  : [batch_size, decoder_targets_length] )
            inference_decode_outputs, inference_final_state, inference_final_sequence_length = \
                        tf.contrib.seq2seq.dynamic_decode(decoder=inference_decoder, maximum_iterations=20)

            self.inference_outputs = inference_decode_outputs.sample_id

    def step(self, sess, data_batch, forward_only, mode):
        if mode == 'train':
            feed_dict = {self.batch_size : data_batch.batch_size,
                         self.encoder_sources : data_batch.encoder_sources, 
                         self.sources_length_list : data_batch.encoder_sources_length_list, 
                         self.decoder_targets : data_batch.decoder_targets, 
                         self.targets_length_list : data_batch.decoder_targets_length_list}
            if forward_only == False:
                train_op, loss = sess.run([self.train_op, self.loss], feed_dict=feed_dict)
                return train_op, loss
            else:
                loss = sess.run(self.loss, feed_dict=feed_dict)
                return loss
        elif mode == 'inference': 
            feed_dict = {self.batch_size : data_batch.batch_size,
                     self.encoder_sources : data_batch.encoder_sources,
                     self.sources_length_list : data_batch.encoder_sources_length_list}
            inference_outputs = sess.run(self.inference_outputs, feed_dict=feed_dict)

            return inference_outputs
        else:
            raise NotImplementedError
