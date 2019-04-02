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
            data_format=None,
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
                internal_layer = BatchNormLayer(current_layer,
                                                is_train=True,
                                                use_cudnn_on_gpu=use_cudnn_on_gpu,
                                                act=tf.nn.relu,
                                                data_format=data_format,
                                                name=("batchnorm_layer" + str(i)))
                internal_layer = Conv2dLayer(internal_layer,
                                             shape=(3, 3, growing_features, growth),
                                             b_init=None,
                                             use_cudnn_on_gpu=use_cudnn_on_gpu,
                                             data_format=data_format,
                                             name=("cnn_layer" + str(i)))
                internal_layer = DropoutLayer(internal_layer,
                                              keep=keep,
                                              data_format=data_format,
                                              use_cudnn_on_gpu=use_cudnn_on_gpu)
                current_layer = ConcatLayer([current_layer, internal_layer],
                                            concat_dim=3,
                                            use_cudnn_on_gpu=use_cudnn_on_gpu,
                                            data_format=data_format,
                                            name="concat_layer" + str(i))
                growing_features += growth

            self.outputs = current_layer.outputs

        self._add_layers(self.outputs)


