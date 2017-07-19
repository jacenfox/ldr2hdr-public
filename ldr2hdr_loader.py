import tensorflow as tf
import os
import numpy as np


class LDR2HDR_Loader():
    '''loads pre-trained model, and grabs the input and output nodes.'''

    def __init__(self, sess=None):
        if sess is None:
            sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        self.sess = sess
        self.model = None

    def load_tf_model(self, tfmodel):
        self.model = tf.train.import_meta_graph(tfmodel)
        self.model.restore(self.sess, tf.train.latest_checkpoint(os.path.split(tfmodel)[0]))
        self.init_input_output()

    def forward(self, input_ims):
        '''
        notice the fist sess.run() will be slow.
        After the graph cached, the subsequent computations will be super faster.
        '''
        self.input_ims = input_ims
        # or you could get the sun elevation directly from the network. See the init_input_output()
        pred, fc = self.sess.run([self.output, self.fc], {self._isTraining: False, self.input: input_ims})
        fc = np.squeeze(fc, axis=[1, 2])
        return pred, fc

    def get_ops(self, name=''):
        '''
        return a tensor by given name, return all if name==''
        '''
        op_names = []
        for op in self.sess.graph.get_operations():
            if (name in op.name and
                    'save' not in op.name and
                    'init' not in op.name and
                    'ema' not in op.name and
                    'Adam' not in op.name and
                    'gradients' not in op.name):
                op_names.append(op.name)
        ops = [self.sess.graph.get_tensor_by_name(op_name + ':0') for op_name in op_names]
        return ops

    def init_input_output(self):
        self.output = self.get_ops('pred_linear')[-1]
        self.fc = self.get_ops('feature/activation')[-1]
        self.sun = self.get_ops('SunPosition/fc5/activation')[-1]  # this branch directly output the sun elevation
        self.input = self.get_ops('InputImage')[0]
        self._isTraining = self.get_ops('isTraining')[0]
