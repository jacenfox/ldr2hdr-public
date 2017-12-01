from ldr2hdr_ops import *
import tensorflow as tf
from gradient_reversal import gradient_reversal


class LDR2HDR_Net():

    def __init__(self, fc_dim, im_height, deconv_method):
        """
        Args:
            doSkipLinks:
            doSigmoidLast: if input is linearHDR, use sigmoid for last layer
            fc_dim: Dimension of fully connect layer. [64]
        """
        self.fc_dim = fc_dim
        self.doSigmoidLast = doSigmoidLast
        self.deconv_method = deconv_method
        self.im_height = im_height
        self.im_width = 2 * im_height

    def encoder(self, inputs, isTraining, reuse=False):
        with tf.variable_scope('Encoders', reuse=reuse):
            fc_dim = self.fc_dim
            ''' encoder '''
            with tf.variable_scope('encoder1'):
                self.conv1 = conv2d(inputs, output_channels=64, k_h=7, k_w=7, pool_method='stride')
                self.conv1 = tf.nn.elu(batch_norm(self.conv1, isTraining), name='activation')

            # Layer2
            with tf.variable_scope('encoder2'):
                self.conv2 = conv2d(self.conv1, output_channels=128, k_h=5, k_w=5, pool_method='stride')
                self.conv2 = tf.nn.elu(batch_norm(self.conv2, isTraining), name='activation')

            # Layer3
            with tf.variable_scope('encoder3'):
                self.conv3 = conv2d(self.conv2, output_channels=256, k_h=3, k_w=3, pool_method='stride')
                self.conv3 = tf.nn.elu(batch_norm(self.conv3, isTraining), name='activation')

            # Layer4
            with tf.variable_scope('encoder4'):
                self.conv4 = conv2d(self.conv3, output_channels=256, k_h=3, k_w=3, pool_method='stride')
                self.conv4 = tf.nn.elu(batch_norm(self.conv4, isTraining), name='activation')

            # Fully connected layer1
            with tf.variable_scope('feature'):
                fc = fc2d(self.conv4, fc_dim, name='fc2d')
                fc = tf.nn.elu(batch_norm(fc, isTraining), name='activation')
                self.fc = fc
            # Apply Dropout
            keep_prob = 0.5
            self.fc = tf.nn.dropout(self.fc, keep_prob, name='dropout')
            return self.fc

    def decoder(self, isTraining, reuse=False):
        fc = self.fc
        conv4 = self.conv4[:, 0:self.im_height / 32, :, :]
        conv3 = self.conv3[:, 0:self.im_height / 16, :, :]
        conv2 = self.conv2[:, 0:self.im_height / 8, :, :]
        conv1 = self.conv1[:, 0:self.im_height / 4, :, :]
        with tf.variable_scope('Decoders', reuse=reuse):
            # de FC layer
            # Reshape fc to fit fully connected layer input
            with tf.variable_scope('defc'):
                defc = dfc2d(fc, out_height=conv4.get_shape()[1].value,
                             out_width=conv4.get_shape()[2].value,
                             out_channels=conv4.get_shape()[3].value, name='dfc_skyonly')
                defc = tf.add(defc, conv4, name='ResidualMatch')
                defc = tf.nn.elu(batch_norm(defc, isTraining), name='activation')
            ''' decoder '''
            with tf.variable_scope('decoder4'):
                deconv4 = deconv2d(defc, output_channels=256, output_imshape=[2 * self.im_height / 32, 2 * self.im_width / 16], k_h=3, k_w=3, method=self.deconv_method)
                deconv4 = tf.add(deconv4, conv3, name='ResidualMatch')
                deconv4 = tf.nn.elu(batch_norm(deconv4, isTraining), name='activation')

            with tf.variable_scope('decoder3'):
                deconv3 = deconv2d(deconv4, output_channels=128, output_imshape=[2 * self.im_height / 16, 2 * self.im_width / 8], k_h=3, k_w=3, method=self.deconv_method)
                deconv3 = tf.add(deconv3, conv2, name='ResidualMatch')
                deconv3 = tf.nn.elu(batch_norm(deconv3, isTraining), name='activation')

            with tf.variable_scope('decoder2'):
                deconv2 = deconv2d(deconv3, output_channels=64, output_imshape=[2 * self.im_height / 8, 2 * self.im_width / 4], k_h=5, k_w=5, method=self.deconv_method)
                deconv2 = tf.add(deconv2, conv1, name='ResidualMatch')
                deconv2 = tf.nn.elu(batch_norm(deconv2, isTraining), name='activation')

            with tf.variable_scope('decoder1'):
                deconv1 = deconv2d(deconv2, output_channels=64, output_imshape=[2 * self.im_height / 4, 2 * self.im_width / 2], k_h=7, k_w=7, method=self.deconv_method, name='decoder1')
                deconv1 = tf.nn.elu(batch_norm(deconv1, isTraining), name='decoder1_activation')

            # Output
            out = conv2d(deconv1, output_channels=3, k_h=1, k_w=1, pool_method=None, name='3ChannelImg')
            out = tf.nn.sigmoid(out, name='OutputImg')
        return out

    def sunPredictior(self, isTraining, reuse=False):
        fc = self.fc
        with tf.variable_scope('SunPosition', reuse=reuse):
            if self.doFCNorFC == 'FC':
                fc_fcn = lambda fc, dims: fc2d(fc, dims)
            elif self.doFCNorFC == 'FCN':
                fc_fcn = lambda fc, dims: conv2d(fc, output_channels=dims, k_h=1, k_w=1, pool_method=None, padding='VALID')

            with tf.variable_scope('fc1'):
                sunpos_fc1 = fc_fcn(fc, 32)
                sunpos_fc1 = tf.nn.elu(batch_norm(sunpos_fc1, isTraining), name='activation')

            with tf.variable_scope('fc2'):
                sunpos_fc2 = fc_fcn(sunpos_fc1, 16)
                sunpos_fc2 = tf.nn.elu(batch_norm(sunpos_fc2, isTraining), name='activation')

            with tf.variable_scope('fc5'):
                sunPos = fc_fcn(sunpos_fc2, 1)
                sunPos = tf.nn.relu(sunPos, name='activation')
                sunPos = tf.squeeze(sunPos, [1, 2], name='output')  # [N,1,1,D]->[N, D] or [N, D]
        return sunPos

    def discriminator_da(self, isTraining, lambdar, reuse=False):
        fc = gradient_reversal(self.fc, lambdar)
        with tf.variable_scope("Discriminator", reuse=reuse):
            with tf.variable_scope('fc1'):
                domain_fc = fc2d(fc, 32)
                domain_fc = tf.nn.elu(domain_fc, name='activation')

            with tf.variable_scope('fc2'):
                domain_logit = fc2d(domain_fc, 2)
                domain_fc = tf.nn.softmax(domain_logit, name='activation')
                domain_fc = tf.squeeze(domain_fc, [1, 2], name='domain_out')
        return domain_fc, domain_logit  # (BATCH, 2)

    def pred(self, inputs, isTraining, reuse=False):
        fc = self.encoder(inputs=inputs, isTraining=isTraining, reuse=reuse)
        sunPos = self.sunPredictior(isTraining=isTraining, reuse=reuse)
        outImg = self.decoder(isTraining=isTraining, reuse=reuse)
        return outImg, sunPos, fc

def usage():
    sess = tf.Session()
    generator = LDR2HDR_Net(fc_dim=64, im_height=64, deconv_method='upsample')
    x = tf.placeholder(tf.float32, [32, 64, 128, 3], name='InputImage')
    outImg, sunPos, fc = generator.pred(inputs=x, isTraining=True)

