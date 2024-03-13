import pandas as pd
from preprocessing import read_test, represent_input_with_features
from tqdm import tqdm
from math import exp
import numpy as np


def q_viterbi(c_word, c_tag, p_word, p_tag, pp_word, pp_tag, n_word, feature2id, pre_trained_weights):
    current_history = (c_word, c_tag, p_word, p_tag, pp_word, pp_tag, n_word)
    relevant_features = represent_input_with_features(history=current_history, dict_of_dicts=feature2id.feature_to_idx)

    exp_current_tag = 0
    exp_all_tags = 0

    for i in relevant_features:
        exp_current_tag += pre_trained_weights[i]

    exp_current_tag = exp(exp_current_tag)

    for tag in feature2id.feature_statistics.tags:
        exp_vect_mult = 0
        demi_history = (c_word, tag, p_word, p_tag, pp_word, pp_tag, n_word)
        demi_hist_relevant_features = represent_input_with_features(history=demi_history,
                                                                    dict_of_dicts=feature2id.feature_to_idx)

        for i in demi_hist_relevant_features:
            exp_vect_mult += pre_trained_weights[i]

        exp_all_tags += exp(exp_vect_mult)

    return exp_current_tag / exp_all_tags


def find_arg_max_k_minus_two(pi, t_u_possible_tags, p_key, k):
    arg_max_k_minus_two = t_u_possible_tags[0][0]

    for pp_key in t_u_possible_tags[:, 0]:
        if (pp_key, p_key) in t_u_possible_tags:

            if pi[(k - 2, pp_key, p_key)] > pi[(k - 2, arg_max_k_minus_two, p_key)]:
                arg_max_k_minus_two = pp_key

    return arg_max_k_minus_two


def compute_probabilities(sentence, pi, bp, all_tags, t_u_possible_tags, feature2id, pre_trained_weights, n, beam):
    for k in range(2, n - 1):
        c_word, p_word, pp_word, n_word = sentence[k], sentence[k - 1], sentence[k - 2], sentence[k + 1]

        for (p_key, c_key) in [(x, y) for y in all_tags for x in t_u_possible_tags[:, 1]]:
            tag_arg_max_k_minus_two = find_arg_max_k_minus_two(pi, t_u_possible_tags, p_key, k)

            pi[(k - 1, p_key, c_key)] = q_viterbi(c_word, c_key, p_word, p_key, pp_word, tag_arg_max_k_minus_two,
                                                  n_word, feature2id, pre_trained_weights) * pi[(k - 2),
                                                                                                tag_arg_max_k_minus_two,
                                                                                                p_key]

            bp[(k - 1, p_key, c_key)] = tag_arg_max_k_minus_two

        filtered_tags_at_time_k = [key for key in pi.keys() if key[0] == k - 1]
        sorted_tags_at_time_k = sorted(filtered_tags_at_time_k, key=lambda key: pi[key], reverse=True)
        top_beam_sorted_tags_at_time_k = np.array(sorted_tags_at_time_k[: beam])
        t_u_possible_tags = np.array([(elem[1], elem[2]) for elem in top_beam_sorted_tags_at_time_k])

    return bp, t_u_possible_tags


def get_tags(bp, t_u_possible_tags, n):
    predicted_tags = [t_u_possible_tags[0][1], t_u_possible_tags[0][0]]

    for k in range(n - 3, 1, -1):
        p_tag = bp[(k, predicted_tags[-1], predicted_tags[-2])]
        predicted_tags.append(p_tag)

    predicted_tags.reverse()

    return predicted_tags


def memm_viterbi(sentence, pre_trained_weights, feature2id):
    """
    Write your MEMM Viterbi implementation below
    You can implement Beam Search to improve runtime
    Implement q efficiently (refer to conditional probability definition in MEMM slides)
    """
    n = len(sentence)

    all_tags = feature2id.feature_statistics.tags.copy()
    all_tags.add('*')

    pi = {(0, '*', '*'): 1}
    bp = {}

    beam = 3
    t_u_possible_tags = np.array([('*', '*')])

    bp, t_u_possible_tags = compute_probabilities(sentence, pi, bp, all_tags, t_u_possible_tags, feature2id,
                                                  pre_trained_weights, n, beam)

    predicted_tags = get_tags(bp, t_u_possible_tags, n)

    return predicted_tags


def tag_all_test(test_path, pre_trained_weights, feature2id, predictions_path):
    tagged = "train" in test_path
    test = read_test(test_path, tagged=tagged)

    output_file = open(predictions_path, "a+")

    for k, sen in tqdm(enumerate(test), total=len(test)):
        sentence = sen[0]
        pred = memm_viterbi(sentence, pre_trained_weights, feature2id)[1:]
        sentence = sentence[2:]
        for i in range(len(pred)):
            if i > 0:
                output_file.write(" ")
            output_file.write(f"{sentence[i]}_{pred[i]}")
        output_file.write("\n")
    output_file.close()


def get_confusion_matrix_and_accuracy(true_tagged_path, predicted_path, feature2id):
    true_tags_with_sentences = read_test(true_tagged_path, True)
    predicted_tags_with_sentences = read_test(predicted_path, True)

    true_tags = []
    predicted_tags = []
    all_tags = set(feature2id.feature_statistics.tags_counts.keys())

    for tuple1, tuple2 in zip(predicted_tags_with_sentences, true_tags_with_sentences):
        predicted_tags.append(tuple1[1][2: -1])
        true_tags.append(tuple2[1][2: -1])

    for tag_sen_pred, tag_sen_true in zip(predicted_tags, true_tags):
        for tag1, tag2 in zip(tag_sen_true, tag_sen_pred):
            if tag1 not in all_tags:
                all_tags.add(tag1)

            if tag2 not in all_tags:
                all_tags.add(tag2)

    all_tags = list(all_tags)
    num_tags = len(all_tags)
    confusion_matrix = np.zeros((num_tags, num_tags))

    dict_tags_index = {all_tags[i]: i for i in range(num_tags)}
    dict_index_tags = {i: all_tags[i] for i in range(num_tags)}

    for tag_sen_pred, tag_sen_true in zip(predicted_tags, true_tags):
        for tag1, tag2 in zip(tag_sen_true, tag_sen_pred):
            true_idx = dict_tags_index[tag1]
            predicted_idx = dict_tags_index[tag2]

            confusion_matrix[true_idx, predicted_idx] += 1

    well_tagged = np.diag(confusion_matrix)
    worst_tags = np.copy(confusion_matrix)

    for i in range(worst_tags.shape[0]):
        worst_tags[i][i] = worst_tags[i][i] - well_tagged[i]

    ten_worst_idx = np.argsort(np.sum(worst_tags, axis=0))[-10:]
    list_worst_tags = []
    tags = []

    for w_idx in ten_worst_idx:
        list_worst_tags.append(dict_index_tags[w_idx])

    for w_idx in dict_index_tags:
        tags.append(dict_index_tags[w_idx])

    w_confusion_matrix = confusion_matrix[ten_worst_idx, :]

    accuracy = 100 * np.trace(confusion_matrix) / np.sum(confusion_matrix)
    w_confusion_matrix = pd.DataFrame(w_confusion_matrix, columns=tags, index=list_worst_tags)

    return w_confusion_matrix, accuracy
