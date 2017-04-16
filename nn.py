import tensorflow as tf
from utils import shared


class HiddenLayer(object):
    """
    Hidden layer with or without bias.
    Input: tensor of dimension (dims*, input_dim)
    Output: tensor of dimension (dims*, output_dim)
    """
    def __init__(self, input_dim, output_dim, bias=True, activation=None,
                 name='hidden_layer'):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bias = bias
        self.name = name
        if activation is None:
            self.activation = None
        elif activation == 'tanh':
            self.activation = tf.nn.tanh
        elif activation == 'sigmoid':
            self.activation = tf.nn.sigmoid
        elif activation == 'softmax':
            self.activation = tf.nn.softmax
        else:
            raise Exception("Unknown activation function: " % activation)

        # Initialize weights and bias
        self.weights = shared((input_dim, output_dim), name + '__weights')
        self.bias = shared((output_dim,), name + '__bias')

    def link(self, input):
        """
        The input has to be a tensor with the right
        most dimension equal to input_dim.
        """
        input_shape = tf.shape(input)
        self.input = tf.reshape(input, (input_shape[0]*input_shape[1], input_shape[-1]))
        self.linear_output = tf.matmul(self.input, self.weights)
        if self.bias:
            self.linear_output = self.linear_output + self.bias
        if self.activation is None:
            self.output = self.linear_output
        else:
            self.output = self.activation(self.linear_output)
        self.output = tf.reshape(self.output, (input_shape[0], input_shape[1], self.output_dim))
        return self.output


class EmbeddingLayer(object):
    """
    Embedding layer: word embeddings representations
    Input: tensor of dimension (dim*) with values in range(0, input_dim)
    Output: tensor of dimension (dim*, output_dim)
    """

    def __init__(self, input_dim, output_dim, name='embedding_layer'):
        """
        Typically, input_dim is the vocabulary size,
        and output_dim the embedding dimension.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.name = name

        # Randomly generate weights
        self.embeddings = shared((input_dim, output_dim),
                                 self.name + '__embeddings')

    def link(self, input):
        """
        Return the embeddings of the given indexes.
        Input: tensor of shape (dim*)
        Output: tensor of shape (dim*, output_dim)
        """
        self.input = input
        self.output = tf.gather(self.embeddings, input)
        return self.output


class DropoutLayer(object):
    """
    Dropout layer. Randomly set to 0 values of the input
    with probability p.
    """
    def __init__(self, p=0.5, name='dropout_layer'):
        """
        p has to be between 0 and 1 (1 excluded).
        p is the probability of dropping out a unit, so
        setting p to 0 is equivalent to have an identity layer.
        """
        assert 0. <= p < 1.
        self.p = p
        self.name = name

    def link(self, input):
        """
        Dropout link: we just apply mask to the input.
        """
        if self.p > 0:
            self.output = tf.nn.dropout(input, 1 - self.p)
        else:
            self.output = input

        return self.output


class LSTM(object):
    """
    Long short-term memory (LSTM). Can be used with or without batches.
    Without batches:
        Input: matrix of dimension (sequence_length, input_dim)
        Output: vector of dimension (output_dim)
    With batches:
        Input: tensor3 of dimension (batch_size, sequence_length, input_dim)
        Output: matrix of dimension (batch_size, output_dim)
    """
    def __init__(self, input_dim, hidden_dim, with_batch=True, name='LSTM'):
        """
        Initialize neural network.
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.with_batch = with_batch
        self.name = name

        # Input gate weights
        self.w_xi = shared((input_dim, hidden_dim), name + '__w_xi')
        self.w_hi = shared((hidden_dim, hidden_dim), name + '__w_hi')
        self.w_ci = shared((hidden_dim, hidden_dim), name + '__w_ci')

        # Forget gate weights
        # self.w_xf = shared((input_dim, hidden_dim), name + '__w_xf')
        # self.w_hf = shared((hidden_dim, hidden_dim), name + '__w_hf')
        # self.w_cf = shared((hidden_dim, hidden_dim), name + '__w_cf')

        # Output gate weights
        self.w_xo = shared((input_dim, hidden_dim), name + '__w_xo')
        self.w_ho = shared((hidden_dim, hidden_dim), name + '__w_ho')
        self.w_co = shared((hidden_dim, hidden_dim), name + '__w_co')

        # Cell weights
        self.w_xc = shared((input_dim, hidden_dim), name + '__w_xc')
        self.w_hc = shared((hidden_dim, hidden_dim), name + '__w_hc')

        # Initialize the bias vectors, c_0 and h_0 to zero vectors
        self.b_i = shared((hidden_dim,), name + '__b_i')
        # self.b_f = shared((hidden_dim,), name + '__b_f')
        self.b_c = shared((hidden_dim,), name + '__b_c')
        self.b_o = shared((hidden_dim,), name + '__b_o')
        self.c_0 = shared((hidden_dim,), name + '__c_0')
        self.h_0 = shared((hidden_dim,), name + '__h_0')

    def link(self, input):
        """
        Propagate the input through the network and return the last hidden
        vector. The whole sequence is also accessible via self.h, but
        where self.h of shape (sequence_length, batch_size, output_dim)
        """
        def recurrence(prev, x_t):
            c_tm1 = prev[0]
            h_tm1 = prev[1]
            if len(x_t.shape) == 1:
                x_t = tf.reshape(x_t, [1, self.input_dim])
            if len(c_tm1.shape) == 1:
                c_tm1 = tf.reshape(c_tm1, [1, self.hidden_dim])
            if len(h_tm1.shape) == 1:
                h_tm1 = tf.reshape(h_tm1, [1, self.hidden_dim])
            i_t = tf.nn.sigmoid(tf.matmul(x_t, self.w_xi) +
                                tf.matmul(h_tm1, self.w_hi) +
                                tf.matmul(c_tm1, self.w_ci) +
                                 self.b_i)
            # f_t = T.nnet.sigmoid(T.dot(x_t, self.w_xf) +
            #                      T.dot(h_tm1, self.w_hf) +
            #                      T.dot(c_tm1, self.w_cf) +
            #                      self.b_f)
            c_t = ((1 - i_t) * c_tm1 + i_t * tf.nn.tanh(tf.matmul(x_t, self.w_xc) +
                                                        tf.matmul(h_tm1, self.w_hc) + self.b_c))
            o_t = tf.nn.sigmoid(tf.matmul(x_t, self.w_xo) +
                                tf.matmul(h_tm1, self.w_ho) +
                                tf.matmul(c_t, self.w_co) +
                                 self.b_o)
            h_t = o_t * tf.nn.tanh(c_t)
            if self.with_batch == False:
                c_t = tf.squeeze(c_t, axis=[0])
                h_t = tf.squeeze(h_t, axis=[0]) 
            return [c_t, h_t]
        # If we use batches, we have to permute the first and second dimension.
        if self.with_batch:
            batch_size = tf.shape(input)[0] 
            zeros = tf.ones([batch_size])
            def alloc(prev, x):
               return [self.c_0, self.h_0]
            out_info = [self.c_0, self.h_0]
            outputs_info = tf.scan(fn=alloc, elems=zeros, initializer=out_info, name='batch_init')
            self.input = tf.transpose(input, (1, 0, 2))
            
        else:
            self.input = input
            outputs_info = [self.c_0, self.h_0]

        states = tf.scan(
            fn=recurrence,
            elems=self.input,
            initializer=outputs_info,
            name='state'
        )
        return states


class GRU(object):
    """
    Gated recurrent unit (GRU). Can be used with or without batches.
    Without batches:
        Input: matrix of dimension (sequence_length, input_dim)
        Output: vector of dimension (output_dim)
    With batches:
        Input: tensor3 of dimension (batch_size, sequence_length, input_dim)
        Output: matrix of dimension (batch_size, output_dim)
    """
    def __init__(self, input_dim, hidden_dim, with_batch=True, name='GRU'):
        """
        Initialize neural network.
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.with_batch = with_batch
        self.name = name

        # Input weight tensor
        self.w_x = shared((input_dim, hidden_dim), name + '__w_x')

        # Reset weight tensor
        self.w_xr = shared((input_dim, hidden_dim), name + '__w_xr')
        self.w_hr = shared((hidden_dim, hidden_dim), name + '__w_hr')

        # Update weight tensor
        self.w_xz = shared((input_dim, hidden_dim), name + '__w_xz')
        self.w_hz = shared((hidden_dim, hidden_dim), name + '__w_hz')

        # Hidden weight tensor
        self.w_h = shared((hidden_dim, hidden_dim), name + '__w_h')

        # Initialize the bias vectors, h_0 to zero vectors
        self.b_r = tf.Variable(tf.truncated_normal((hidden_dim,), mean=1), name=(name + '__b_r'))
        self.b_z = tf.Variable(tf.truncated_normal((hidden_dim,), mean=1), name=(name + '__b_z'))
        self.h_0 = shared((hidden_dim,), name + '__h_0')

    def link(self, input):
        """
        Propagate the input through the network and return the last hidden
        vector. The whole sequence is also accessible via self.h, but
        where self.h of shape (sequence_length, batch_size, output_dim)
        """

        def recurrence(previous_hidden_state, x_t):
            if len(x_t.shape) == 1:
                x_t = tf.reshape(x_t, [1, self.input_dim])
            if len(previous_hidden_state.shape) == 1:
                previous_hidden_state = tf.reshape(previous_hidden_state, [1, self.hidden_dim])

            # update gate
            z_t = tf.sigmoid(tf.matmul(x_t, self.w_xz) + tf.matmul(previous_hidden_state, self.w_hz) + self.b_z)
            # reset gate
            r_t = tf.sigmoid(tf.matmul(x_t, self.w_xr) + tf.matmul(previous_hidden_state, self.w_hr) + self.b_r)
            # candidate activation
            h_ = tf.tanh(tf.matmul(x_t, self.w_x) + tf.matmul(tf.multiply(previous_hidden_state, r_t), self.w_h))

            h_t = tf.multiply((1 - z_t), previous_hidden_state) + tf.multiply(z_t, h_)

            if self.with_batch == False:
                h_t = tf.squeeze(h_t, axis=[0])
            return h_t

        # If we use batches, we have to permute the first and second dimension.
        if self.with_batch:
            batch_size = tf.shape(input)[0]
            zeros = tf.ones([batch_size])

            def alloc(prev, x):
                return self.h_0

            out_info = self.h_0
            outputs_info = tf.scan(fn=alloc, elems=zeros, initializer=out_info, name='batch_init')
            self.input = tf.transpose(input, (1, 0, 2))

        else:
            self.input = input
            outputs_info = self.h_0

        states = tf.scan(
            fn=recurrence,
            elems=self.input,
            initializer=outputs_info,
            name='state'
        )
        return states


def log_sum_exp(x, axis=None):
    """
    Sum probabilities in the log-space.
    """
    xmax = tf.reduce_max(x, axis=axis, keep_dims=True)
    xmax_ = tf.reduce_max(x, axis=axis)
    return xmax_ + tf.log(tf.reduce_sum(tf.exp(x - xmax), axis=axis))


def get_array_arg_max_coordinate(x):
    """
    Get the coodinate of the max score in each row of the matrix x
    :param x: matrix
    :return: coodinate
    """
    shape_x = tf.shape(x)
    row_size = shape_x[0]
    row_size_range = tf.range(0, row_size)
    row_size_range_reshape = tf.reshape(row_size_range, (row_size, 1))
    x_reshape = tf.reshape(x, (row_size, 1))
    row_argmax_coodinate = tf.concat([row_size_range_reshape, x_reshape], axis=1)
    return row_argmax_coodinate


def forward_batch(observations, transitions, viterbi=False,
            return_alpha=False, return_best_sequence=False):
    """
    Takes as input:
        - observations, sequence of shape (batch_size, n_steps, n_classes)
        - transitions, sequence of shape (n_classes, n_classes)
    Probabilities must be given in the log space.
    Compute alpha, matrix of size (n_steps, batch_size n_classes), such that
    alpha[i, j] represents one of these 2 values:
        - the probability that the real path at node i ends in j
        - the maximum probability of a path finishing in j at node i (Viterbi)
    Returns one of these 2 values:
        - alpha
        - the final probability, which can be:
            - the sum of the probabilities of all paths
            - the probability of the best path (Viterbi)
    """
    assert not return_best_sequence or (viterbi and not return_alpha)

    shape_t = transitions.get_shape().dims
    transitions_ = tf.reshape(transitions, (1, shape_t[0].value, shape_t[1].value))

    def recurrence(prev, obs):
        previous = prev
        if return_best_sequence:
            previous = prev[0]
        shape_ = tf.shape(previous)
        previous = tf.reshape(previous, (shape_[0], shape_t[0].value, 1))
        obs = tf.reshape(obs, (shape_[0], 1, shape_t[0].value))
        if viterbi:
            scores = previous + obs + transitions_
            out = tf.reduce_max(scores, axis=1)
            if return_best_sequence:
                out2 = tf.argmax(scores, axis=1)
                return [out, out2]
            else:
                return out
        else:
            return log_sum_exp(previous + obs + transitions, axis=1)

    obs = tf.transpose(observations, (1, 0, 2))
    initial = obs[0]
    ones = tf.ones(tf.shape(initial), dtype=tf.int64)
    if return_best_sequence:
        initial = [initial, ones]
    alpha = tf.scan(
        fn=recurrence,
        elems=obs[1:],
        initializer=initial
    )
    if return_alpha:
        return alpha
    elif return_best_sequence:
        output_info = get_array_arg_max_coordinate(tf.cast(tf.argmax(alpha[0][-1], axis=1), tf.int32))

        def recurrence_cal(prev, x):
            sequ = tf.gather_nd(x, prev)
            return get_array_arg_max_coordinate(sequ)
        sequence = tf.scan(
            fn=recurrence_cal,
            elems=tf.cast(alpha[1][::-1], tf.int32),
            initializer=output_info
        )
        sequence = sequence[:, :, -1]
        sequence = tf.concat([sequence[::-1], [tf.cast(tf.argmax(alpha[0][-1], axis=1), tf.int32)]], axis=0)
        return tf.transpose(sequence)
    else:
        if viterbi:
            return tf.reduce_max(alpha[-1], axis=1)
        else:
            return log_sum_exp(alpha[-1], axis=1)


def forward(observations, transitions, viterbi=False,
            return_alpha=False, return_best_sequence=False):
    """
    Takes as input:
        - observations, sequence of shape (n_steps, n_classes)
        - transitions, sequence of shape (n_classes, n_classes)
    Probabilities must be given in the log space.
    Compute alpha, matrix of size (n_steps, n_classes), such that
    alpha[i, j] represents one of these 2 values:
        - the probability that the real path at node i ends in j
        - the maximum probability of a path finishing in j at node i (Viterbi)
    Returns one of these 2 values:
        - alpha
        - the final probability, which can be:
            - the sum of the probabilities of all paths
            - the probability of the best path (Viterbi)
    """
    assert not return_best_sequence or (viterbi and not return_alpha)

    def recurrence(prev, obs):
        previous = prev
        if return_best_sequence:
            previous = prev[0]
        previous = tf.expand_dims(previous, 1)
        obs = tf.expand_dims(obs, 0)
        if viterbi:
            scores = previous + obs + transitions
            out = tf.reduce_max(scores, axis=0)
            if return_best_sequence:
                out2 = tf.argmax(scores, axis=0)
                return [out, out2]
            else:
                return out
        else:
            return log_sum_exp(previous + obs + transitions, axis=0)

    
    initial = observations[0]
    ones = tf.ones(tf.shape(initial), dtype=tf.int64)
    if return_best_sequence:
        initial = [initial, ones]
    alpha = tf.scan(
        fn=recurrence,
        elems=observations[1:],
        initializer=initial
    )
    if return_alpha:
        return alpha
    elif return_best_sequence:
        output_info = tf.cast(tf.argmax(alpha[0][-1], axis=0), tf.int32)
        sequence = tf.scan(
            fn=lambda previous, beta_i: beta_i[previous], 
            elems=tf.cast(alpha[1][::-1], tf.int32),
            initializer=output_info
        )
        sequence = tf.concat([sequence[::-1], [tf.cast(tf.argmax(alpha[0][-1], axis=0), tf.int32)]], axis=0)
        return sequence    
    else:
        if viterbi:
            return tf.reduce_max(alpha[-1], axis=0)
        else:
            return log_sum_exp(alpha[-1], axis=0)
