import tensorflow as tf
import tensorlayer as tl

from layers.denseblock import DenseBlock

tf.logging.set_verbosity(tf.logging.DEBUG)
tl.logging.set_verbosity(tl.logging.DEBUG)

densenet_264_structure = dict(block1=6, block2=12, block3=64, block4=48,
                              k=24, n_units=1000)


def model_densenet(placeholder_x, placeholder_y_, reuse,
                   keep=0.8,
                   densenet_bc=True,
                   is_train=True,
                   structure=densenet_264_structure,
                   use_cudnn_on_gpu=False):
    if densenet_bc is True:
        compression = True
        bottle_neck = True
    else:
        compression = False
        bottle_neck = False

    with tf.variable_scope("densenet_imagenet", reuse=reuse):

        net = tl.layers.InputLayer(placeholder_x, name='input')
        net = tl.layers.Conv2dLayer(net,
                                    shape=(7, 7, 3, 2 * structure['k']),
                                    strides=(1, 2, 2, 1),
                                    data_format='NHWC',
                                    b_init=None,
                                    name='conv_in')

        net = tl.layers.PoolLayer(net,
                                  ksize=(1, 3, 3, 1),
                                  strides=(1, 2, 2, 1),
                                  pool=tf.nn.max_pool,
                                  name='maxpool_in'
                                  )
        features = 2 * structure['k']

        for idx in range(3):
            net = DenseBlock(net,
                             block_layers=structure['block' + str(idx + 1)],
                             in_features=features,
                             growth=structure['k'],
                             keep=keep,
                             is_train=is_train,
                             bottle_neck=bottle_neck,
                             use_cudnn_on_gpu=use_cudnn_on_gpu,
                             name='denseblock' + str(idx + 1)
                             )
            features = net.output_features

            if compression is True:
                out_features = features // 2
            else:
                out_features = features

            net = tl.layers.BatchNormLayer(net,
                                           act=tf.nn.relu,
                                           is_train=is_train,
                                           name="batch_norm" + str(idx + 1)
                                           )
            net = tl.layers.Conv2dLayer(net,
                                        shape=(1, 1, features, out_features),
                                        b_init=None,
                                        data_format='NHWC',
                                        use_cudnn_on_gpu=use_cudnn_on_gpu,
                                        name='conv' + str(idx + 1))
            net = tl.layers.PoolLayer(net,
                                      pool=tf.nn.avg_pool,
                                      name='pool_layer' + str(idx + 1))

            features = out_features

        net = DenseBlock(net,
                         block_layers=structure['block4'],
                         in_features=features,
                         growth=structure['k'],
                         keep=keep,
                         is_train=is_train,
                         bottle_neck=bottle_neck,
                         use_cudnn_on_gpu=use_cudnn_on_gpu,
                         name='denseblock4'
                         )

        features = net.output_features

        net = tl.layers.PoolLayer(net,
                                  ksize=(1, 7, 7, 1),
                                  strides=(1, 7, 7, 1),
                                  pool=tf.nn.avg_pool,
                                  name='pool_layer_out')

        net = tl.layers.ReshapeLayer(net,
                                     [-1, features])
        net = tl.layers.DenseLayer(net,
                                   n_units=structure['n_units'],
                                   name='output_layer')

        y = net.outputs
        y_op = tf.argmax(tf.nn.softmax(y), 1)

        cost = tl.cost.cross_entropy(y, placeholder_y_, name='ce')

        return net


if __name__ == '__main__':
    sess = tf.InteractiveSession()

    x = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name='x')
    y_ = tf.placeholder(tf.int64, shape=[None], name='y_')

    net = model_densenet(x, y_)

    sess.run(tf.global_variables_initializer())

    net.print_params()
