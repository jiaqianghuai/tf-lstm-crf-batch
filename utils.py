import re
import datetime
import numpy as np
import tensorflow as tf


models_path = "./models"
models_saver_path = "./models/saver"


def get_name(parameters):
    """
    Generate a model name from its parameters.
    """
    l = []
    for k, v in parameters.items():
        if type(v) is str and "/" in v:
            l.append((k, v[::-1][:v[::-1].index('/')][::-1]))
        else:
            l.append((k, v))
    name = ",".join(["%s=%s" % (k, str(v).replace(',', '')) for k, v in l])
    return "".join(i for i in name if i not in "\/:*?<>|")


def shared(shape, name):
    """
    Create a shared object of a numpy array.
    """
    if len(shape) == 1:
        # bias are initialized with zeros
        return tf.get_variable(name, shape, tf.float32, tf.constant_initializer(0))
    else:
        drange = np.sqrt(6. / (np.sum(shape)))
        return tf.get_variable(name, shape, tf.float32, tf.random_uniform_initializer(-drange, drange))

def create_dico(item_list):
    """
    Create a dictionary of items from a list of list of items.
    """
    assert type(item_list) is list
    dico = {}
    for items in item_list:
        for item in items:
            if item not in dico:
                dico[item] = 1
            else:
                dico[item] += 1
    return dico


def create_mapping(dico):
    """
    Create a mapping (item to ID / ID to item) from a dictionary.
    Items are ordered by decreasing frequency.
    """
    sorted_items = sorted(dico.items(), key=lambda x: (-x[1], x[0]))
    id_to_item = {i: v[0] for i, v in enumerate(sorted_items)}
    item_to_id = {v: k for k, v in id_to_item.items()}
    return item_to_id, id_to_item


def zero_digits(s):
    """
    Replace every digit in a string by a zero.
    """
    return re.sub('\d', '0', s)


def insert_singletons(words, singletons, p=0.5):
    """
    Replace singletons by the unknown word with a probability p.
    """
    new_words = []
    for word in words:
        if word in singletons and np.random.uniform() < p:
            new_words.append(0)
        else:
            new_words.append(word)
    return new_words


def pad_word_chars(words):
    """
    Pad the characters of the words in a sentence.
    Input:
        - list of lists of ints (list of words, a word being a list of char indexes)
    Output:
        - padded list of lists of ints
        - padded list of lists of ints (where chars are reversed)
        - list of ints corresponding to the index of the last character of each word
    """
    max_length = max([len(word) for word in words])
    char_for = []
    char_rev = []
    char_pos = []
    for word in words:
        padding = [0] * (max_length - len(word))
        char_for.append(word + padding)
        char_rev.append(word[::-1] + padding)
        char_pos.append(len(word) - 1)
    return char_for, char_rev, char_pos


def pad_word_chars(words, max_length):
    """
    Pad the characters of the words in a sentence.
    Input:
        - list of lists of ints (list of words, a word being a list of char indexes)
        - the max length of word
    Output:
        - padded list of lists of ints
        - padded list of lists of ints (where chars are reversed)
        - list of ints corresponding to the index of the last character of each word
    """
    char_for = []
    char_rev = []
    char_pos = []
    for word in words:
        padding = [0] * (max_length - len(word))
        char_for.append(word + padding)
        char_rev.append(word[::-1] + padding)
        char_pos.append(len(word) - 1)
    return char_for, char_rev, char_pos


def pad_sentence_words(sentences):
    """
    Pad the words of the sentence in the batch_sentence.
    Input:
        - list of lists of ints (list of sentence, a sentence being a list of word indexes)
    Output:
        - padded list of lists of ints
        - padded list of lists of ints (where chars are reversed)
        - list of ints corresponding to the index of the last character of each word
    """
    max_length = max([len(sentence) for sentence in sentences])
    word_for = []
    word_pos = []
    for words in sentences:
        padding = [0] * (max_length - len(words))
        word_for.append(words + padding)
        word_pos.append(len(words) - 1)
    return word_for, word_pos


def create_input(data, parameters, add_label, singletons=None):
    """
    Take sentence data and return an input for
    the training or the evaluation function.
    """
    words = data['words']
    chars = data['chars']
    if singletons is not None:
        words = insert_singletons(words, singletons)
    if parameters['cap_dim']:
        caps = data['caps']
    char_for, char_rev, char_pos = pad_word_chars(chars)
    input = []
    if parameters['word_dim']:
        input.append(words)
    if parameters['char_dim']:
        input.append(char_for)
        if parameters['char_bidirect']:
            input.append(char_rev)
        input.append(char_pos)
    if parameters['cap_dim']:
        input.append(caps)
    if add_label:
        input.append(data['tags'])
    return input


def create_input_(data, n_tags, parameters, add_label, singletons=None):
    """
    Take sentence data and return an input for
    the training or the evaluation function.
    """
    words = data['words']
    chars = data['chars']
    if singletons is not None:
        words = insert_singletons(words, singletons)
    char_for, char_rev, char_pos = pad_word_chars(chars)
    input = []
    input.append(words)
    input.append(char_for)
    input.append(char_rev)
    char_pos_array = []
    for i in xrange(len(char_pos)):
        temp = []
        temp.append(i)
        temp.append(char_pos[i])
        char_pos_array.append(temp)
    input.append(char_pos_array)
    if add_label:
        input.append(data['tags'])
        tag_id_trans_array = []
        tag_len = len(data['tags'])
        for i in xrange(tag_len):
            temp = []
            if i == 0:
                temp.append(n_tags)
                temp.append(data['tags'][i])
                tag_id_trans_array.append(temp)
            temp = []
            if i < (tag_len - 1):
                temp.append(data['tags'][i])
                temp.append(data['tags'][i+1])
                tag_id_trans_array.append(temp)
            else:
                temp.append(data['tags'][i])
                temp.append(n_tags+1)
                tag_id_trans_array.append(temp)
        input.append(tag_id_trans_array)
        tag_index_array = []
        for i in xrange(tag_len):
            temp = []
            temp.append(i)
            temp.append(data['tags'][i])
            tag_index_array.append(temp)
        input.append(tag_index_array)
    return input


def create_input_batch(sentences, parameters, n_tags=0, add_label=False, singletons=None):
    """
    Take batch_sentence data and return a batch_input for
    the training or the evaluation function.
    """
    batch_size = len(sentences)
    max_length = max([len(sentence['words']) for sentence in sentences])
    #print 'max_length..................................'
    #print max_length
    input = []
    words_batch = []
    chars_batch = []
    chars_max_batch = []
    input_tag = []
    input_tag_id_trans = []
    input_tag_id_index = []
    for k in xrange(batch_size):
        data = sentences[k]
        words = data['words']
        words_len = len(words)
        #print 'words........................'
        #print words
        if singletons is not None:
            words = insert_singletons(words, singletons)
        words_batch.append(words)
        if parameters['char_dim']:
            chars = data['chars']
            max_word_length = max([len(word) for word in chars])
            chars_max_batch.append(max_word_length)
            if len(chars) > words_len:
                chars = chars[0:words_len]
            assert len(chars) == words_len
            if len(chars) < max_length:
                padding = [[0]] * (max_length - len(chars))
                chars.extend(padding)
            chars_batch.append(chars)
        if add_label:
            #print 'tags............................'
            #print data['tags']
            tag_len = len(data['tags'])
            if tag_len > words_len:
                data['tags'] = data['tags'][0:words_len]
            tag_len = len(data['tags'])
            assert words_len == tag_len
            tag_array = data['tags']
            #print tag_array
            if max_length > tag_len:
                for i in xrange(max_length - tag_len):
                    tag_array.append(0)
            input_tag.append(tag_array)
            tag_id_trans_array = []
            #print 'tag_len.........................'
            #print tag_len
            for i in xrange(tag_len):
                temp = []
                if i == 0:
                    temp.append(n_tags)
                    temp.append(data['tags'][i])
                    tag_id_trans_array.append(temp)
                temp = []
                if i < (tag_len - 1):
                    temp.append(data['tags'][i])
                    temp.append(data['tags'][i+1])
                    tag_id_trans_array.append(temp)
                else:
                    temp.append(data['tags'][i])
                    temp.append(n_tags+1)
                    tag_id_trans_array.append(temp)
            if max_length > tag_len:
                for i in xrange(max_length - tag_len):
                    temp = []
                    if i == 0:
                        temp.append(n_tags+1)
                        temp.append(0)
                        tag_id_trans_array.append(temp)
                    else:
                        temp.append(0)
                        temp.append(0)
                        tag_id_trans_array.append(temp)
            input_tag_id_trans.append(tag_id_trans_array)
            tag_index_array = []
            for i in xrange(tag_len):
                temp = []
                temp.append(i)
                temp.append(data['tags'][i])
                tag_index_array.append(temp)
            if max_length > tag_len:
                for i in xrange(max_length - tag_len):
                    temp = []
                    temp.append(i+tag_len)
                    temp.append(0)
                    tag_index_array.append(temp)
            input_tag_id_index.append(tag_index_array)
    # words
    words_for, words_pos = pad_sentence_words(words_batch)
    input.append(words_for)
    input.append(words_pos)
    input_char_for = []
    input_char_rev = []
    if parameters['char_dim']:
        chars_max_batch_all = max(chars_max_batch)
        count = 0
        char_pos_array = []
        for i in xrange(len(chars_batch)):
            chars = chars_batch[i]
            char_for, char_rev, char_pos = pad_word_chars(chars, chars_max_batch_all)
            input_char_for.append(char_for)
            input_char_rev.append(char_rev)
            for pos in char_pos:
                temp = []
                temp.append(count)
                temp.append(pos)
                char_pos_array.append(temp)
                count += 1
        input.append(input_char_for)
        input.append(input_char_rev)
        input.append(char_pos_array)
    if add_label:
        batch_input_tag_id_trans = np.vstack([np.expand_dims(x, 0) for x in input_tag_id_trans])
        batch_input_tag_id_index = np.vstack([np.expand_dims(x, 0) for x in input_tag_id_index])
        input.append(input_tag)
        input.append(batch_input_tag_id_trans)
        input.append(batch_input_tag_id_index)
    return input


def evaluate(sess, f_eval, model, parameters, parsed_sentences, n_tags=0):
    count = 0
    token_accus_all = []
    sentence_accus_all = []
    batch_size = parameters['batch_size']
    start_time = datetime.datetime.now()
    token_count = 0.0
    while count < len(parsed_sentences):
        batch_data = []
        for i in xrange(batch_size):
            index = i + count
            if index >= len(parsed_sentences):
                index %= len(parsed_sentences)
            data = parsed_sentences[index]
            batch_data.append(parsed_sentences[index])
        input_ = create_input_batch(batch_data, parameters, n_tags, True)
        feed_dict_ = {}
        if parameters['char_dim']:
            feed_dict_[model.word_ids] = input_[0]
            feed_dict_[model.word_pos_ids] = input_[1]
            feed_dict_[model.char_for_ids] = input_[2]
            feed_dict_[model.char_rev_ids] = input_[3]
            feed_dict_[model.char_pos_ids] = input_[4]
            input_tag = input_[5]
        else:
            feed_dict_[model.word_ids] = input_[0]
            feed_dict_[model.word_pos_ids] = input_[1]
            input_tag = input_[2]
        f_scores = sess.run(f_eval, feed_dict=feed_dict_)
        accus_batch = []
        sentence_batch = []
        if parameters['crf']:
            for x in xrange(len(batch_data)):
                f_score = f_scores[x]
                word_pos = input_[1][x] + 2
                y_pred = f_score[1:word_pos]
                y_real = input_tag[x][0:(word_pos-1)]
                correct_prediction = np.equal(y_pred, y_real)
                accus = np.array(correct_prediction).astype(float).sum()
                accus_mean = np.array(correct_prediction).astype(float).mean()
                accus_batch.append(accus)
                if accus_mean < 1.0:
                    sentence_batch.append(0.0)
                else:
                    sentence_batch.append(1.0)
                token_count += (input_[1][x] + 1)
            accus_val = accus_batch
            sentence_val = np.array(sentence_batch).astype(float).mean()
        else:
            y_preds = f_scores.argmax(axis=-1)
            y_reals = np.array(input_tag).astype(np.int32)
            for x in xrange(batch_size):
                word_pos = input_[1][x] + 1
                y_pred = y_preds[x][0:word_pos]
                y_real = y_reals[x][0:word_pos]
                correct_prediction = np.equal(y_pred, y_real)
                accus = np.array(correct_prediction).astype(float).sum()
                accus_mean = np.array(correct_prediction).astype(float).mean()
                accus_batch.append(accus)
                if accus_mean < 1.0:
                    sentence_batch.append(0.0)
                else:
                    sentence_batch.append(1.0)
                token_count += word_pos
            accus_val = accus_batch
            sentence_val = np.array(sentence_batch).astype(float).mean()
        count += batch_size
        token_accus_all.extend(accus_val)
        sentence_accus_all.append(sentence_val)
    token_accuracy = np.sum(token_accus_all) / (token_count + 0.000001)
    sentence_accuracy = np.mean(sentence_accus_all)
    end_time = datetime.datetime.now()
    cost_time = (end_time - start_time).seconds
    print "token accuracy: %f, sentence accuracy: %f, cost time: %i" % (token_accuracy, sentence_accuracy,  cost_time)
    return token_accuracy, sentence_accuracy
