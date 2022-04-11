import numpy as np
import tensorflow as tf

ALPHA = 0.5

def restrictedSoftmax(a):
    exp_a = tf.math.exp(ALPHA * a)
    sum_exp_a = tf.convert_to_tensor(tf.reduce_sum(exp_a))
    y = exp_a / sum_exp_a
    return y


if __name__ == '__main__':
    test = [1, 2, 3, 4, 5]
    test = np.array(test)

    print(restrictedSoftmax(test))