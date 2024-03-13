import pickle
import time
from preprocessing import preprocess_train
from preprocessing_small_model import cross_validation
from optimization import get_optimal_vector


def main():
    """Initialization of the hyperparameters"""
    threshold_big_model, lam_big_model = 1, 1
    threshold_small_model, lam_small_model = 1, 1

    train_path_big_model = "data/train1.wtag"
    train_path_small_model = "data/train2.wtag"

    weights_path_big_model = 'weights.pkl'
    weights_path_small_model = 'weights_small_model.pkl'

    predictions_path_small_model = 'predictions_small_model.wtag'

    """-----------------------------------------------------------------------------------------------------------------
       Big Model"""

    print("Big Model: \n")

    beginning = time.time()

    statistics, feature2id = preprocess_train(train_path_big_model, threshold_big_model)
    get_optimal_vector(statistics=statistics, feature2id=feature2id, weights_path=weights_path_big_model,
                       lam=lam_big_model)

    with open(weights_path_big_model, 'rb') as f:
        optimal_params, feature2id = pickle.load(f)
    pre_trained_weights = optimal_params[0]

    end = time.time()
    print(pre_trained_weights)
    print(f"Training the model took {end - beginning:.2f}s\n")

    """-----------------------------------------------------------------------------------------------------------------
    Small Model"""

    print("Small Model: \n")

    cross_validation(train_path_small_model, weights_path_small_model, predictions_path_small_model,
                     threshold_small_model, lam_small_model, 10)

    beginning = time.time()

    statistics, feature2id = preprocess_train(train_path_small_model, threshold_small_model)
    get_optimal_vector(statistics=statistics, feature2id=feature2id, weights_path=weights_path_small_model,
                       lam=lam_small_model)

    with open(weights_path_small_model, 'rb') as f:
        optimal_params, feature2id = pickle.load(f)
    pre_trained_weights = optimal_params[0]

    end = time.time()
    print(pre_trained_weights)
    print(f"Training the model took {end - beginning:.2f}s\n")


if __name__ == '__main__':
    main()
