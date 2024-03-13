import pickle
from inference import tag_all_test


def main():
    # Initialisation of the hyperparameters
    test_path_big_model = "data/comp1.words"
    test_path_small_model = "data/comp2.words"

    weights_path_big_model = 'weights.pkl'
    weights_path_small_model = 'weights_small_model.pkl'

    predictions_path_big_model = 'comp_m1_342791324_931214522.wtag'
    predictions_path_small_model = 'comp_m2_342791324_931214522.wtag'

    # Big Model

    with open(weights_path_big_model, 'rb') as f:
        optimal_params, feature2id = pickle.load(f)
    pre_trained_weights = optimal_params[0]

    tag_all_test(test_path_big_model, pre_trained_weights, feature2id, predictions_path_big_model)

    # Small Model

    with open(weights_path_small_model, 'rb') as f:
        optimal_params, feature2id = pickle.load(f)
    pre_trained_weights = optimal_params[0]

    tag_all_test(test_path_small_model, pre_trained_weights, feature2id, predictions_path_small_model)


if __name__ == '__main__':
    main()
