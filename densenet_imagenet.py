import tensorflow as tf
import tensorlayer as tl

from layers.denseblock import DenseBlock

tf.logging.set_verbosity(tf.logging.DEBUG)
tl.logging.set_verbosity(tl.logging.DEBUG)


densenet_264_structure = dict(block_1=6, block_2=12, block_3=64, block_4=48, k=24)


def model_densenet(placeholder_x, placeholder_y_, keep=0.8, structure=densenet_264_structure, use_cudnn_on_gpu=False):
    with tf.variable_scope("densenet_imagenet"):
        # 224*224
        net = tl.layers.InputLayer(placeholder_x, name='input')
        net = tl.layers.Conv2dLayer(net,
                                    shape=(7, 7, 3, 16),
                                    strides=(1, 2, 2, 1),
                                    b_init=None,
                                    name='conv_in')
        # 112*112
        net = tl.layers.PoolLayer(net,
                                  ksize=(1, 3, 3, 1),
                                  strides=(1, 2, 2, 1),
                                  pool=tf.nn.max_pool,
                                  name='maxpool_in'
                                  )
        # 56*56
        net = DenseBlock(net,
                         block_layers=structure['block_1'],
                         in_features=16,
                         growth=structure['k'],
                         keep=keep,
                         use_cudnn_on_gpu=use_cudnn_on_gpu,
                         name='denseblock1'
                         )

        # 56*56
        features = net.output_features
        net = tl.layers.BatchNormLayer(net,
                                       act=tf.nn.relu,
                                       is_train=True,
                                       name="batch_norm1"
                                       )
        net = tl.layers.Conv2dLayer(net,
                                    shape=(1, 1, features, features),
                                    b_init=None,
                                    use_cudnn_on_gpu=use_cudnn_on_gpu,
                                    name='conv1')
        net = tl.layers.PoolLayer(net,
                                  pool=tf.nn.avg_pool,
                                  name='pool_layer1')
        # 28*28
        net = DenseBlock(net,
                         block_layers=structure['block_2'],
                         in_features=features,
                         growth=structure['k'],
                         keep=keep,
                         use_cudnn_on_gpu=use_cudnn_on_gpu,
                         name='denseblock2'
                         )

        features = net.output_features
        net = tl.layers.BatchNormLayer(net,
                                       act=tf.nn.relu,
                                       is_train=True,
                                       name="batch_norm2"
                                       )
        net = tl.layers.Conv2dLayer(net,
                                    shape=(1, 1, features, features),
                                    b_init=None,
                                    use_cudnn_on_gpu=use_cudnn_on_gpu,
                                    name='conv2')
        net = tl.layers.PoolLayer(net,
                                  pool=tf.nn.avg_pool,
                                  name='pool_layer2')
        # 14*14

        net = DenseBlock(net,
                         block_layers=structure['block_3'],
                         in_features=features,
                         growth=structure['k'],
                         keep=keep,
                         use_cudnn_on_gpu=use_cudnn_on_gpu,
                         name='denseblock3'
                         )

        features = net.output_features
        net = tl.layers.BatchNormLayer(net,
                                       act=tf.nn.relu,
                                       is_train=True,
                                       name="batch_norm3"
                                       )
        net = tl.layers.Conv2dLayer(net,
                                    shape=(1, 1, features, features),
                                    b_init=None,
                                    use_cudnn_on_gpu=use_cudnn_on_gpu,
                                    name='conv3')
        net = tl.layers.PoolLayer(net,
                                  pool=tf.nn.avg_pool,
                                  name='pool_layer2')
        # 7*7
        net = tl.layers.PoolLayer(net,
                                  ksize=(1, 7, 7, 1),
                                  strides=(1, 7, 7, 1),
                                  pool=tf.nn.avg_pool,
                                  name='pool_layer3')

        net = tl.layers.ReshapeLayer(net,
                                     [-1, features])
        net = tl.layers.DenseLayer(net,
                                   n_units=10,
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

    net.print_layers()
