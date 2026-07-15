import numpy as np

from llm_spectral_dynamics.controls import dimension_shuffle, time_shuffle_sequences, token_order_shuffle


def test_time_shuffle_preserves_sequence_shape_and_values():
    arr = np.arange(2 * 5 * 3).reshape(2, 5, 3)
    shuffled = time_shuffle_sequences(arr, seed=2)
    assert shuffled.shape == arr.shape
    for i in range(arr.shape[0]):
        assert sorted(map(tuple, shuffled[i])) == sorted(map(tuple, arr[i]))


def test_dimension_shuffle_preserves_column_marginals():
    arr = np.arange(20).reshape(10, 2)
    shuffled = dimension_shuffle(arr, seed=3)
    for j in range(arr.shape[1]):
        assert sorted(shuffled[:, j].tolist()) == sorted(arr[:, j].tolist())


def test_token_order_shuffle_preserves_first_token_when_requested():
    ids = np.arange(12).reshape(3, 4)
    shuffled = token_order_shuffle(ids, seed=4, preserve_first=True)
    np.testing.assert_array_equal(shuffled[:, 0], ids[:, 0])
    for i in range(ids.shape[0]):
        assert sorted(shuffled[i, 1:].tolist()) == sorted(ids[i, 1:].tolist())

