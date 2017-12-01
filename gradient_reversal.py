import tensorflow as tf
from tensorflow.python.framework import ops


class FlipGradientBuilder(object):
    """
    Gradient Reversal Layer.
    Discussion:
        https://github.com/fchollet/keras/issues/3119
        https://github.com/tensorflow/tensorflow/issues/4342
    Code from here:
        https://github.com/rwth-i6/returnn/blob/master/TFUtil.py#L273-L299
        https://github.com/pumpikano/tf-dann/blob/master/flip_gradient.py
    """

    def __init__(self):
        self.num_calls = 0

    def __call__(self, x, lambdar=1.0):
        grad_name = "FlipGradient%d" % self.num_calls

        @ops.RegisterGradient(grad_name)
        def _flip_gradients(op, grad):
            if tf.__version__[0:3] == '1.2':
                return [tf.negative(grad) * lambdar]
            else:
                return [tf.neg(grad) * lambdar]

        g = tf.get_default_graph()
        with g.gradient_override_map({"Identity": grad_name}):
            y = tf.identity(x)

        self.num_calls += 1
        return y


gradient_reversal = FlipGradientBuilder()
