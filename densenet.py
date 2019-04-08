import tensorflow as tf
import tensorlayer as tl

from layers.denseblock import DenseBlock

tf.logging.set_verbosity(tf.logging.DEBUG)
tl.logging.set_verbosity(tl.logging.DEBUG)


def model_densenet(placeholder_x, placeholder_y_, keep=0.8, use_cudnn_on_gpu=False):
    with tf.variable_scope("densenet_cifar10"):
        net = tl.layers.InputLayer(placeholder_x, name='input')
        net = tl.layers.Conv2dLayer(net,
                                    shape=(3, 3, 3, 16),
                                    b_init=None,
                                    name='conv_in')
        net = DenseBlock(net,
                         block_layers=12,
                         in_features=16,
                         growth=12,
                         keep=keep,
                         use_cudnn_on_gpu=use_cudnn_on_gpu,
                         name='denseblock1'
                         )

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
        net = DenseBlock(net,
                         block_layers=12,
                         in_features=features,
                         growth=12,
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
        net = DenseBlock(net,
                         block_layers=12,
                         in_features=features,
                         growth=12,
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
        net = tl.layers.PoolLayer(net,
                                  ksize=(1, 8, 8, 1),
                                  strides=(1, 8, 8, 1),
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

    X_train, y_train, X_test, y_test = tl.files.load_cifar10_dataset(shape=(-1, 32, 32, 3))

    tl.prepro.threading_data()

    sess = tf.InteractiveSession()

    x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name='x')
    y_ = tf.placeholder(tf.int64, shape=[None], name='y_')

    net = model_densenet(x, y_)

    sess.run(tf.global_variables_initializer())

    net.print_layers()