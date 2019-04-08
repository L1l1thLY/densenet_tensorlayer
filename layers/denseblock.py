import tensorflow as tf

from tensorlayer.layers.core import Layer

from tensorlayer import logging

from tensorlayer.layers.normalization import BatchNormLayer
from tensorlayer.layers.convolution.expert_conv import Conv2dLayer
from tensorlayer.layers.dropout import DropoutLayer
from tensorlayer.layers import ConcatLayer


class DenseBlock(Layer):
    def __init__(
            self,
            prev_layer,
            block_layers,
            in_features,
            growth,
            keep,
            use_cudnn_on_gpu=None,
            bottle_neck=True,
            name='dense_block',
    ):
        super(DenseBlock, self).__init__(prev_layer=prev_layer, name=name)

        logging.info(
            "DenseBlock %s: layers: %d growth(k): %d dropout layer keep: %d act: %s" % (
                self.name,
                block_layers,
                growth,
                keep,
                self.act.__name__ if self.act is not None else 'No Activation'
            )
        )

        with tf.variable_scope(name):
            current_layer = prev_layer
            growing_features = in_features

            for i in range(block_layers):
                # When bottleneck is unused,
                # internal in features is always same with growing_features
                internal_in_features = growing_features

                if bottle_neck is True:
                    internal_layer = BatchNormLayer(current_layer,
                                                    is_train=True,
                                                    act=tf.nn.relu,
                                                    name=("batchnorm_layer_bn" + str(i + 1)))
                    internal_layer = Conv2dLayer(internal_layer,
                                                 shape=(1, 1, internal_in_features, 4 * growth),
                                                 b_init=None,
                                                 use_cudnn_on_gpu=use_cudnn_on_gpu,
                                                 name=("cnn_layer_bn" + str(i + 1)))
                    internal_in_features = 4 * growth

                internal_layer = BatchNormLayer(internal_layer if bottle_neck else current_layer,
                                                is_train=True,
                                                act=tf.nn.relu,
                                                name=("batchnorm_layer" + str(i + 1)))
                internal_layer = Conv2dLayer(internal_layer,
                                             shape=(3, 3, internal_in_features, growth),
                                             b_init=None,
                                             use_cudnn_on_gpu=use_cudnn_on_gpu,
                                             name=("cnn_layer" + str(i + 1)))
                internal_layer = DropoutLayer(internal_layer,
                                              keep=keep)

                current_layer = ConcatLayer([current_layer, internal_layer],
                                            concat_dim=3,
                                            name="concat_layer" + str(i + 1))
                growing_features += growth

            self.outputs = current_layer.outputs

        self.output_features = growing_features
        self._add_layers(self.outputs)


