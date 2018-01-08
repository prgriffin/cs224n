import numpy as np


def softmax(x):
    """Compute the softmax function for each row of the input x.

    It is crucial that this function is optimized for speed because
    it will be used frequently in later code. You might find numpy
    functions np.exp, np.sum, np.reshape, np.max, and numpy
    broadcasting useful for this task.

    Numpy broadcasting documentation:
    http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html

    You should also make sure that your code works for a single
    N-dimensional vector (treat the vector as a single row) and
    for M x N matrices. This may be useful for testing later. Also,
    make sure that the dimensions of the output match the input.

    You must implement the optimization in problem 1(a) of the
    written assignment!

    Arguments:
    x -- A N dimensional vector or M x N dimensional numpy matrix.

    Return:
    x -- You are allowed to modify x in-place
    """
    orig_shape = x.shape

    if len(x.shape) > 1:
        # Matrix
        print("x = ", x)
        amax = np.amax(x, axis=1)
        amax = np.reshape(amax, (amax.shape[0], 1))
        normalized = x - amax
        print("normalized = ", normalized)
        e_to_x = np.exp(normalized)
        total = np.sum(e_to_x, axis=1)
        x = e_to_x / np.reshape(total, (total.shape[0], 1))
    else:
        # Vector
        amax = np.amax(x)
        normalized = x - amax
        e_to_x = np.exp(normalized)
        total = np.sum(e_to_x)
        x = e_to_x / total

    assert x.shape == orig_shape
    return x


def test_softmax_basic():
    """
    Some simple tests to get you started.
    Warning: these are not exhaustive.
    """
    print("Running basic tests...")
    test1 = softmax(np.array([1, 2]))
    print(test1)
    ans1 = np.array([0.26894142,  0.73105858])
    assert np.allclose(test1, ans1, rtol=1e-05, atol=1e-06)

    test2 = softmax(np.array([[1001,1002],[3,4]]))
    print(test2)
    ans2 = np.array([
        [0.26894142, 0.73105858],
        [0.26894142, 0.73105858]])
    assert np.allclose(test2, ans2, rtol=1e-05, atol=1e-06)

    test3 = softmax(np.array([[-1001,-1002]]))
    print(test3)
    ans3 = np.array([0.73105858, 0.26894142])
    assert np.allclose(test3, ans3, rtol=1e-05, atol=1e-06)

    print("You should be able to verify these results by hand!\n")


def test_softmax():
    """
    Use this space to test your softmax implementation by running:
        python q1_softmax.py
    This function will not be called by the autograder, nor will
    your tests be graded.
    """
    print("Running your tests...")
    test1 = np.linspace(-1000, 1000, 20)
    test2 = np.linspace(-20, 20, 20)
    result1 = softmax(test1)
    result2 = softmax(test2)

    array1 = np.array((test1, test2))
    result3 = softmax(array1)
    assert np.allclose(result1, result3[0, :])
    assert np.allclose(result2, result3[1, :])


if __name__ == "__main__":
    test_softmax_basic()
    test_softmax()
