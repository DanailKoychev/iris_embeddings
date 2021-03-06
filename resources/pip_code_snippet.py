import numpy

# may need to adjust the below function if it is not working correctly for your w2v model
def make_fingerprint_matrix(w2v_model, concepts):
    vectors = [w2v_model.wv[c] if c in w2v_model.wv else numpy.zeros(w2v_model.vector_size) for c in concepts]
    fingerprint_matrix = numpy.array([v for v in vectors])
    return fingerprint_matrix

def calculate_pip_distance_of_arrays(x, y):

    x = x/numpy.linalg.norm(x)
    y = y/numpy.linalg.norm(y)

    x_tr = numpy.transpose(x)
    y_tr = numpy.transpose(y)

    pip_x = numpy.dot(x, x_tr)
    pip_y = numpy.dot(y, y_tr)

    pip_loss = numpy.linalg.norm((pip_x - pip_y))
    return pip_loss


def calculate_pip_distance(reference_words_list, reference_word_model_matrix, w2v_model_to_compare):
	# for reference_words_list use the words from ref_vocab.json
    words_matrix_1 = reference_word_model_matrix # load using pickle python module from file ref_vocab_matrix.pkl
    words_matrix_2 = make_fingerprint_matrix(w2v_model_to_compare, reference_words_list)
    return calculate_pip_distance_of_arrays(words_matrix_1, words_matrix_2)
