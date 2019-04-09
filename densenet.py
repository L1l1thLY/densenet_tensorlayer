import tensorflow as tf
import tensorlayer as tl
import numpy as np
import time

from layers.denseblock import DenseBlock

tf.logging.set_verbosity(tf.logging.DEBUG)
tl.logging.set_verbosity(tl.logging.DEBUG)

X_train, y_train, X_test, y_test = tl.files.load_cifar10_dataset(shape=(-1, 32, 32, 3))


def distort_fn(x, is_train=False):
    # x = tl.prepro.crop(x, 32, 32, is_random=is_train)
    x = (x - np.mean(x)) / max(np.std(x), 1e-5)  # avoid values divided by 0
    return x


def model_densenet(placeholder_x, placeholder_y_, reuse,
                   is_train=True,
                   keep=0.8,
                   densenet_bc=True,
                   use_cudnn_on_gpu=False):
    if densenet_bc is True:
        compression = True
        bottle_neck = True
    else:
        compression = False
        bottle_neck = False

    with tf.variable_scope("densenet_cifar10", reuse=reuse):
        model_net = tl.layers.InputLayer(placeholder_x, name='input')
        model_net = tl.layers.Conv2dLayer(model_net,
                                          shape=(3, 3, 3, 16),
                                          b_init=None,
                                          data_format='NHWC',
                                          name='conv_in')

        features = 16
        for idx in range(2):
            model_net = DenseBlock(model_net,
                                   block_layers=12,
                                   in_features=features,
                                   growth=12,
                                   keep=keep,
                                   bottle_neck=bottle_neck,
                                   is_train=is_train,
                                   use_cudnn_on_gpu=use_cudnn_on_gpu,
                                   name='denseblock' + str(idx + 1)
                                   )

            features = model_net.output_features

            if compression is True:
                out_features = features // 2
            else:
                out_features = features

            model_net = tl.layers.BatchNormLayer(model_net,
                                                 act=tf.nn.relu,
                                                 is_train=is_train,
                                                 name='batch_norm' + str(idx + 1)
                                                 )
            model_net = tl.layers.Conv2dLayer(model_net,
                                              shape=(1, 1, features, out_features),
                                              b_init=None,
                                              data_format='NHWC',
                                              use_cudnn_on_gpu=use_cudnn_on_gpu,
                                              name='conv' + str(idx + 1))
            model_net = tl.layers.PoolLayer(model_net,
                                            pool=tf.nn.avg_pool,
                                            name='pool_layer' + str(idx + 1))

            features = out_features

        model_net = DenseBlock(model_net,
                               block_layers=12,
                               in_features=features,
                               growth=12,
                               keep=keep,
                               bottle_neck=bottle_neck,
                               is_train=is_train,
                               use_cudnn_on_gpu=use_cudnn_on_gpu,
                               name='denseblock3'
                               )

        features = model_net.output_features
        model_net = tl.layers.BatchNormLayer(model_net,
                                             act=tf.nn.relu,
                                             is_train=is_train,
                                             name="batch_norm3"
                                             )
        model_net = tl.layers.PoolLayer(model_net,
                                        ksize=(1, 8, 8, 1),
                                        strides=(1, 8, 8, 1),
                                        pool=tf.nn.avg_pool,
                                        name='pool_layer3')
        model_net = tl.layers.ReshapeLayer(model_net,
                                           [-1, features])
        model_net = tl.layers.DenseLayer(model_net,
                                         n_units=10,
                                         name='output_layer')
        y = model_net.outputs
        y_op = tf.argmax(tf.nn.softmax(y), 1)

        cost = tl.cost.cross_entropy(y, placeholder_y_, name='ce')

        correct = tf.equal(y_op, y_)
        acc = tf.reduce_mean(tf.cast(correct, tf.float32))

        return model_net, cost, acc


if __name__ == '__main__':
    sess = tf.InteractiveSession()

    x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name='x')
    y_ = tf.placeholder(tf.int64, shape=[None], name='y_')

    net, cost, _ = model_densenet(x, y_, reuse=False, is_train=True, use_cudnn_on_gpu=True)
    _, cost_test, acc = model_densenet(x, y_, True, is_train=False, use_cudnn_on_gpu=True)

    n_epoch = 50000
    learning_rate = 0.0001
    print_freq = 1
    batch_size = 128

    train_params = net.all_params
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost, var_list=train_params)

    sess.run(tf.global_variables_initializer())

    for epoch in range(n_epoch):
        start_time = time.time()
        for X_train_a, y_train_a in tl.iterate.minibatches(X_train, y_train, batch_size, shuffle=True):
            X_train_a = tl.prepro.threading_data(X_train_a, fn=distort_fn, is_train=True)  # data augmentation for training
            sess.run(train_op, feed_dict={x: X_train_a, y_: y_train_a})

        if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
            print("Epoch %d of %d took %fs" % (epoch + 1, n_epoch, time.time() - start_time))

            test_loss, test_acc, n_batch = 0, 0, 0
            for X_test_a, y_test_a in tl.iterate.minibatches(X_test, y_test, batch_size, shuffle=False):
                X_test_a = tl.prepro.threading_data(X_test_a, fn=distort_fn, is_train=False)  # central crop
                err, ac = sess.run([cost_test, acc], feed_dict={x: X_test_a, y_: y_test_a})
                test_loss += err
                test_acc += ac
                n_batch += 1

            print("   test loss: %f" % (test_loss / n_batch))
            print("   test acc: %f" % (test_acc / n_batch))

