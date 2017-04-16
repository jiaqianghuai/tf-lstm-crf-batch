import os
import numpy as np
import tensorflow as tf
import cPickle

from utils import shared, get_name
from nn import HiddenLayer, EmbeddingLayer, LSTM, forward


class Model(object):
    """
    Network architecture.
    """
    def __init__(self, parameters=None, models_path=None, model_path=None):
        """
        Initialize the model. We either provide the parameters and a path where
        we store the models, or the location of a trained model.
        """
        if model_path is None:
            assert parameters and models_path
            # Create a name based on the parameters
            self.parameters = parameters
            self.name = get_name(parameters)
            # Model location
            model_path = os.path.join(models_path, self.name)
            self.model_path = model_path
            self.mappings_path = os.path.join(model_path, 'mappings.pkl')
            self.parameters_path = os.path.join(model_path, 'parameters.pkl')
            # Create directory for the model if it does not exist
            if not os.path.exists(self.model_path):
                os.makedirs(self.model_path)
            # Save the parameters to disk
            with open(self.parameters_path, 'wb') as f:
                cPickle.dump(parameters, f)
        else:
            assert parameters is None and models_path is None
            # Model location
            self.model_path = model_path
            self.mappings_path = os.path.join(model_path, 'mappings.pkl')
            self.parameters_path = os.path.join(model_path, 'parameters.pkl')
            # Load the parameters and the mappings from disk
            with open(self.parameters_path, 'rb') as f:
                self.parameters = cPickle.load(f)
            self.reload_mappings()

    def save_mappings(self, id_to_word, id_to_char, id_to_tag):
        """
        We need to save the mappings if we want to use the model later.
        """
        self.id_to_word = id_to_word
        self.id_to_char = id_to_char
        self.id_to_tag = id_to_tag
        with open(self.mappings_path, 'wb') as f:
            mappings = {
                'id_to_word': self.id_to_word,
                'id_to_char': self.id_to_char,
                'id_to_tag': self.id_to_tag,
            }
            cPickle.dump(mappings, f)

    def reload_mappings(self):
        """
        Load mappings from disk.
        """
        with open(self.mappings_path, 'rb') as f:
            mappings = cPickle.load(f)
        self.id_to_word = mappings['id_to_word']
        self.id_to_char = mappings['id_to_char']
        self.id_to_tag = mappings['id_to_tag']

    def build(self,
              dropout,
              char_dim,
              char_lstm_dim,
              char_bidirect,
              word_dim,
              word_lstm_dim,
              word_bidirect,
              lr_method,
              lr_rate,
              clip_norm,
              crf,
              is_train,
              **kwargs
              ):
        """
        Build the network.
        """
        # Training parameters
        n_words = len(self.id_to_word)
        n_chars = len(self.id_to_char)
        n_tags = len(self.id_to_tag)

        # Network variables
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None], name='word_ids') # shape:[batch_size, max_word_len]
        self.word_pos_ids = tf.placeholder(tf.int32, shape=[None], name='word_pos_ids') # shape: [batch_size]
        self.char_for_ids = tf.placeholder(tf.int32, shape=[None, None, None], name='char_for_ids') # shape: [batch_size, word_max_len, char_max_len]
        self.char_rev_ids = tf.placeholder(tf.int32, shape=[None, None, None], name='char_rev_ids') # shape: [batch_size, word_max_len, char_max_len]
        self.char_pos_ids = tf.placeholder(tf.int32, shape=[None, None], name='char_pos_ids') # shape: [batch_size*word_max_len, char_max_len]
        self.tag_ids = tf.placeholder(tf.int32, shape=[None, None], name='tag_ids') # shape: [batch_size,word_max_len]
        self.tag_id_trans = tf.placeholder(tf.int32, shape=[None, None, None], name='tag_id_trans') # shape: [batch_size,word_max_len+1,2] 
        self.tag_id_index = tf.placeholder(tf.int32, shape=[None, None, None], name='tag_id_index') # shape: [batch_size,word_max_len,2]
        # Final input (all word features)
        input_dim = 0
        inputs = []
        #
        # Word inputs
        #
        if word_dim:
            input_dim += word_dim
            with tf.device("/cpu:0"):
                word_layer = EmbeddingLayer(n_words, word_dim, name='word_layer')
                word_input = word_layer.link(self.word_ids)
                inputs.append(word_input)
        
        #
        # Phars inputs
        #
        if char_dim:
            input_dim += char_lstm_dim
            char_layer = EmbeddingLayer(n_chars, char_dim, name='char_layer')

            char_lstm_for = LSTM(char_dim, char_lstm_dim, with_batch=True,
                                 name='char_lstm_for')
            char_lstm_rev = LSTM(char_dim, char_lstm_dim, with_batch=True,
                                 name='char_lstm_rev')

            with tf.device("/cpu:0"):
                char_for_embedding_batch = char_layer.link(self.char_for_ids)
                char_rev_embedding_batch = char_layer.link(self.char_rev_ids)
            shape_for = tf.shape(char_for_embedding_batch)
            # reshape from [batch_size, word_max_len, char_max_len, char_dim] to [batch_size*word_max_len, char_max_len, char_dim]
            char_for_embedding = tf.reshape(char_for_embedding_batch,
                                            (shape_for[0]*shape_for[1], shape_for[2], shape_for[3]))
            shape_rev = tf.shape(char_rev_embedding_batch)
            char_rev_embedding = tf.reshape(char_rev_embedding_batch,
                                            (shape_rev[0] * shape_rev[1], shape_rev[2], shape_rev[3]))
            char_lstm_for_states = char_lstm_for.link(char_for_embedding)
            char_lstm_rev_states = char_lstm_rev.link(char_rev_embedding)
            char_lstm_for_h_trans = tf.transpose(char_lstm_for_states[1], (1, 0, 2), name='char_lstm_for_h_trans')
            char_lstm_rev_h_trans = tf.transpose(char_lstm_rev_states[1], (1, 0, 2), name='char_lstm_rev_h_trans')
            char_for_output = tf.gather_nd(char_lstm_for_h_trans, self.char_pos_ids, name='char_for_output')
            char_rev_output = tf.gather_nd(char_lstm_rev_h_trans, self.char_pos_ids, name='char_rev_output')
            char_for_output_batch = tf.reshape(char_for_output, (shape_for[0], shape_for[1], char_lstm_dim))
            char_rev_output_batch = tf.reshape(char_rev_output, (shape_rev[0], shape_rev[1], char_lstm_dim))
            inputs.append(char_for_output_batch)
            if char_bidirect:
                inputs.append(char_rev_output_batch)
                input_dim += char_lstm_dim
        inputs = tf.concat(inputs, axis=-1)
        # Dropout on final input
        assert dropout < 1 and 0.0 <= dropout
        if dropout:
            input_train = tf.nn.dropout(inputs, 1 - dropout)
            if is_train:
                inputs = input_train 
        # LSTM for words
        word_lstm_for = LSTM(input_dim, word_lstm_dim, with_batch=True,
                             name='word_lstm_for')
        word_lstm_rev = LSTM(input_dim, word_lstm_dim, with_batch=True,
                             name='word_lstm_rev')
        # fordword hidden output
        word_states_for = word_lstm_for.link(inputs)
        word_lstm_for_output = tf.transpose(word_states_for[1], (1, 0, 2), name='word_lstm_for_h_trans')

        # reverse hidden ouput
        inputs_rev = tf.reverse_sequence(inputs, self.word_pos_ids, seq_dim=1, batch_dim=0)
        word_states_rev = word_lstm_rev.link(inputs_rev)
        word_lstm_rev_h_trans = tf.transpose(word_states_rev[1], (1, 0, 2), name='word_lstm_rev_h_trans')
        word_lstm_rev_output = tf.reverse_sequence(word_lstm_rev_h_trans, self.word_pos_ids, seq_dim=1, batch_dim=0)
        if word_bidirect:
            final_output = tf.concat([word_lstm_for_output, word_lstm_rev_output],axis=-1)
            tanh_layer = HiddenLayer(2 * word_lstm_dim, word_lstm_dim, name='tanh_layer', activation='tanh')
            final_output = tanh_layer.link(final_output)
        else:
            final_output = word_lstm_for_output
        final_layer = HiddenLayer(word_lstm_dim, n_tags, name='final_layer')
        tags_scores = final_layer.link(final_output)
        # No CRF
        if not crf:
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.tag_ids, logits=tags_scores, name='xentropy')
            cost = tf.reduce_mean(cross_entropy, name='xentropy_mean')
        else:
            transitions = shared((n_tags + 2, n_tags + 2), 'transitions')
            small = -1000
            b_s = np.array([[small] * n_tags + [0, small]]).astype(np.float32)
            e_s = np.array([[small] * n_tags + [small, 0]]).astype(np.float32)

            # for batch observation
            #def recurrence(prev, obs):
            #    s_len = tf.shape(obs)[0]
            #    obvs = tf.concat([obs, small * tf.ones((s_len, 2))], axis=1)
            #    observations = tf.concat([b_s, obvs, e_s], axis=0)
            #    return observations
            #tags_scores_shape = tf.shape(tags_scores)
            #obs_initial = tf.ones((tags_scores_shape[1] + 2, n_tags + 2))
            #obs_batch = tf.scan(fn=recurrence, elems=tags_scores, initializer=obs_initial)
            
            # Score from tags
            def recurrence_real_score(prev,obs):
                tags_score = obs[0]
                tag_id_index_ = obs[1]
                tag_id_trans_= obs[2] 
                word_pos_ = obs[3] + 1
                tags_score_slice = tags_score[0:word_pos_,:]
                tag_id_index_slice = tag_id_index_[0:word_pos_,:]
                tag_id_trans_slice = tag_id_trans_[0:(word_pos_+1),:]
                real_path_score = tf.reduce_sum(tf.gather_nd(tags_score_slice, tag_id_index_slice))
                real_path_score += tf.reduce_sum(tf.gather_nd(transitions, tag_id_trans_slice))
                return tf.reshape(real_path_score,[])
            real_path_score_list = tf.scan(fn=recurrence_real_score, elems=[tags_scores, self.tag_id_index, self.tag_id_trans, self.word_pos_ids], initializer=0.0)
            
            def recurrence_all_path(prev, obs):
                tags_score = obs[0]
                word_pos_ = obs[1] + 1
                tags_score_slice = tags_score[0:word_pos_,:]
                s_len = tf.shape(tags_score_slice)[0]
                obvs = tf.concat([tags_score_slice, small * tf.ones((s_len, 2))], axis=1)
                observations = tf.concat([b_s, obvs, e_s], axis=0)
                all_paths_scores = forward(observations, transitions)
                return tf.reshape(all_paths_scores,[])
            all_paths_scores_list = tf.scan(fn=recurrence_all_path, elems=[tags_scores, self.word_pos_ids], initializer=0.0)
            cost = - tf.reduce_mean(real_path_score_list - all_paths_scores_list)
        # Network parameters
        if not crf:
            f_score = tf.nn.softmax(tags_scores)
        else:
            def recurrence_predict(prev, obs):
                tags_score = obs[0]
                word_pos_ = obs[1] + 1
                tags_score_slice = tags_score[0:word_pos_,:]
                s_len = tf.shape(tags_score_slice)[0]
                obvs = tf.concat([tags_score_slice, small * tf.ones((s_len, 2))], axis=1)
                observations = tf.concat([b_s, obvs, e_s], axis=0)
                all_paths_scores = forward(observations, transitions, viterbi=True, return_alpha=False, return_best_sequence=True)
                all_paths_scores = tf.concat([all_paths_scores, tf.zeros([tf.shape(tags_score)[0]-s_len], tf.int32)], axis=0)
                return all_paths_scores
            f_score = tf.scan(fn=recurrence_predict, elems=[tags_scores, self.word_pos_ids], initializer=tf.zeros([tf.shape(tags_scores)[1]+2], tf.int32)) 
        # Optimization
        tvars = tf.trainable_variables()
        grads = tf.gradients(cost, tvars)
        if clip_norm > 0:
            grads, _ = tf.clip_by_global_norm(grads, clip_norm)
        
        if lr_method == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(lr_rate)
        elif lr_method == 'adagrad':
            optimizer = tf.train.AdagradOptimizer(lr_rate)
        elif lr_method == 'adadelta':
            optimizer = tf.train.AdadeltaOptimizer(lr_rate)
        elif lr_method == 'adam':
            optimizer = tf.train.AdamOptimizer(lr_rate)
        elif lr_method == 'rmsprop':
            optimizer = tf.train.RMSPropOptimizer(lr_rate)
        else:
            raise("Not implemented learning method: %s" % lr_method)
        
        train_op = optimizer.apply_gradients(zip(grads, tvars))

        return cost, f_score, train_op
