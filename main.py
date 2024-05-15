import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras import backend as K

from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask
from loaddata import load_cifar, load_mnist, load_medmnist, load_fashionmnist, load_chex, load_ret, load_blood
from plothistory import save_plothistory
from conf_matrix import plot_save_confmatrix



K.set_image_data_format('channels_last')




def CapsNet(input_shape, n_class, routings):
    """
    A Capsule Network on MNIST.
    :param input_shape: data shape, 3d, [width, height, channels]
    :param n_class: number of classes
    :param routings: number of routing iterations
    :param batch_size: size of batch
    :return: Two Keras Models, the first one used for training, and the second one for evaluation.
            `eval_model` can also be used for training.
    """


    x = layers.Input(shape=input_shape)
    x1 = res_block(x, filters=32)
    x1 = layers.BatchNormalization()(x1)


    conv1 = layers.Conv2D(filters=32, kernel_size=5, strides=1, padding='valid', activation='relu', name='conv1')(x1)
    conv2 = layers.Conv2D(filters=64, kernel_size=5, strides=1, padding='valid', activation='relu', name='conv2')(conv1)

    # Line2
    conv3 = layers.Conv2D(filters=64, kernel_size=9, strides=1, padding='valid', activation='relu', name='conv3')(x1)

    # concat Line1 and Line2 --> Line12
    conva = layers.Concatenate()([conv2, conv3])

    conva = layers.BatchNormalization()(conva)

    # conv= layers.Conv2D(filters=64, kernel_size=9, strides=2, padding='valid', activation='relu', name='conv')(conva)

    # Line12-1
    conv4 = layers.Conv2D(filters=64, kernel_size=5, strides=1, padding='valid', activation='relu', name='conv4')(
        conva)
    conv5 = layers.Conv2D(filters=128, kernel_size=5, strides=1, padding='valid', activation='relu', name='conv5')(
        conv4)

    # Line12-2
    conv6 = layers.Conv2D(filters=128, kernel_size=9, strides=1, padding='valid', activation='relu', name='conv6')(
        conva)

    # Line3
    conv7 = layers.Conv2D(filters=128, kernel_size=17, strides=1, padding='valid', activation='relu', name='conv7')(x1)

    # concat Line12-1, Line12-2 and Line3 --> Line123
    convb = layers.Concatenate()([conv5, conv6, conv7])

    convb= layers.BatchNormalization()(convb)

    # conv8 = layers.Conv2D(filters=64, kernel_size=11, strides=2, padding='valid', activation='relu', name='conv8')(convb)

    convb1d = layers.Conv2D(filters=128, kernel_size=1, strides=2, padding='valid', activation='relu', name='convb1d')(
        convb)
    # Line123-1
    conv9 = layers.Conv2D(filters=256, kernel_size=5, strides=1, padding='valid', activation='relu', name='conv9')(
        convb1d)
    conv10 = layers.Conv2D(filters=256, kernel_size=5, strides=1, padding='valid', activation='relu', name='conv10')(
        conv9)

    # Line123-2
    conv11 = layers.Conv2D(filters=256, kernel_size=9, strides=1, padding='valid', activation='relu', name='conv11')(
        convb1d)

    # concat Line123-1 and Line123-2 --> Line12312
    convc = layers.Concatenate()([conv10, conv11])
    convc = layers.BatchNormalization()(convc)

    # Line12312-1
    conv12 = layers.Conv2D(filters=256, kernel_size=5, strides=1, padding='valid', activation='relu', name='conv12')(
        convc)
    conv13 = layers.Conv2D(filters=512, kernel_size=5, strides=1, padding='valid', activation='relu', name='conv13')(
        conv12)

    # Line12312-2
    conv14 = layers.Conv2D(filters=512, kernel_size=9, strides=1, padding='valid', activation='relu', name='conv14')(
        convc)

    # Line123-3
    conv15 = layers.Conv2D(filters=512, kernel_size=17, strides=1, padding='valid', activation='relu', name='conv15')(
        convb1d)

    # Line4
    conv16 = layers.Conv2D(filters=512, kernel_size=11, strides=3, padding='valid', activation='relu', name='conv16')(x1)

    # concat Line1-2 and Line3
    convd = layers.Concatenate()([conv16, conv15, conv14, conv13])

    convd = layers.BatchNormalization()(convd)

    # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]

    primarycaps = PrimaryCap(convd, dim_capsule=8, n_channels=32, kernel_size=9, strides=2, padding='valid')
    #primarycaps= layers.BatchNormalization()(primarycaps)

    #primarycaps = layers.BatchNormalization()(primarycaps)
    # Layer 3: Capsule layer. Routing algorithm works here.
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=16, routings=routings, name='digitcaps')(primarycaps)

    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    # If using tensorflow, this will not be necessary. :)
    out_caps = Length(name='capsnet')(digitcaps)

    # Decoder network.
    y = layers.Input(shape=(n_class,))
    masked_by_y = Mask()([digitcaps, y])  # The true label is used to mask the output of capsule layer. For training

    masked = Mask()(digitcaps)  # Mask using the capsule with maximal length. For prediction

    # Shared Decoder model in training and prediction
    decoder = models.Sequential(name='decoder')
    decoder.add(layers.Dense(512, activation='relu', input_dim=16* n_class))
    decoder.add(layers.Dense(1024, activation='relu'))
    decoder.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))
    decoder.add(layers.Reshape(target_shape=input_shape, name='out_recon'))

    # Models for training and evaluation (prediction)
    train_model = models.Model([x, y], [out_caps, decoder(masked_by_y)])
    eval_model = models.Model(x, [out_caps, decoder(masked)])


    return train_model, eval_model


def margin_loss(y_true, y_pred):
    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """
    # return tf.reduce_mean(tf.square(y_pred))
    L = y_true * tf.square(tf.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * tf.square(tf.maximum(0., y_pred - 0.1))

    return tf.reduce_mean(tf.reduce_sum(L, 1))


def res_block(input_tensor, filters, kernel_size=3, stride=1):
    x = layers.Conv2D(filters, kernel_size=kernel_size, strides=stride, padding='same', activation='relu')(input_tensor)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(filters, kernel_size=kernel_size, strides=stride, padding='same')(x)
    x = layers.BatchNormalization()(x)
    # Skip connection (Residual connection)
    input_tensor = layers.Conv2D(filters, kernel_size=1, strides=stride, padding='same')(input_tensor)
    x = layers.Add()([x, input_tensor])
    x = layers.Activation('relu')(x)
    x = layers.BatchNormalization()(x)
    return x


def train(model,  # type: models.Model
          data, args):
    """
    Training a CapsuleNet
    :param model: the CapsuleNet model
    :param data: a tuple containing training and testing data, like `((x_train, y_train), (x_test, y_test))`
    :param args: arguments
    :return: The trained model
    """
    # unpacking the data
   # (x_train, y_train), (x_val, y_val) = data
    (x_train, y_train), (x_test, y_test) = data
    # callbacks
    log = callbacks.CSVLogger(args.save_dir + '/log.csv')
    checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}.h5', monitor='val_capsnet_acc',
                                           save_best_only=True, save_weights_only=True, verbose=1)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (args.lr_decay ** epoch))

    # compile the model

    model.compile(optimizer=optimizers.Adam(lr=args.lr),
                  loss=[margin_loss, 'mse'],
                  loss_weights=[1., args.lam_recon],
                  metrics={'capsnet': 'accuracy'})


    # Begin: Training with data augmentation ---------------------------------------------------------------------#
    def train_generator(x, y, batch_size, shift_fraction=0.):
        train_datagen = ImageDataGenerator(width_shift_range=shift_fraction,
                                           height_shift_range=shift_fraction)  # shift up to 2 pixel for MNIST
        generator = train_datagen.flow(x, y, batch_size=batch_size)
        while 1:
            x_batch, y_batch = generator.next()
            yield (x_batch, y_batch), (y_batch, x_batch)

    # Training with data augmentation. If shift_fraction=0., no augmentation.

    import datetime
    start=datetime.datetime.now()
    #model.load_weights('result/base_model/trained_model.h5') --- model yükleme
    #early = EarlyStopping(monitor='val_loss', mode='min', patience=20)
    caps= model.fit(train_generator(x_train, y_train, args.batch_size, args.shift_fraction),
              steps_per_epoch=int(y_train.shape[0] / args.batch_size),
              epochs=args.epochs,
              validation_data=([x_test, y_test], [y_test, x_test]), batch_size=args.batch_size,
              callbacks=[log, checkpoint, lr_decay])
    # End: Training with data augmentation -----------------------------------------------------------------------

    end=datetime.datetime.now()
    print("Zaman: ", end-start)

    model.save_weights('result/base_model.h5')

    df=pd.DataFrame([(end-start)])
    df.to_csv('result/time.csv')

    return caps


def test(model, data, args):
    from sklearn.metrics import accuracy_score, auc
    x_test, y_test = data
    y_pred, x_recon = model.predict(x_test, batch_size=32)
    prediction=np.argmax(y_pred, axis=1)
    y_true=np.argmax(y_test, axis=1)
    test_acc = np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1)) / y_test.shape[0]
    accuracy_s=accuracy_score(y_true, prediction)
    auc = tf.keras.metrics.AUC(num_thresholds=16)(y_test, y_pred).numpy()
    plot_save_confmatrix(y_test=y_true, y_pred=prediction)
    return test_acc,prediction, accuracy_s, auc


if __name__ == "__main__":
    import os
    import argparse
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras import callbacks

    # setting the hyper parameters
    parser = argparse.ArgumentParser(description="Capsule Network on MNIST.")
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--lr', default=0.0001, type=float,
                        help="Initial learning rate")
    parser.add_argument('--lr_decay', default=0.9, type=float,
                        help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")
    parser.add_argument('--lam_recon', default=0.392, type=float,
                        help="The coefficient for the loss of decoder")
    parser.add_argument('-r', '--routings', default=3, type=int,
                        help="Number of iterations used in routing algorithm. should > 0")
    parser.add_argument('--shift_fraction', default=0.1, type=float,
                        help="Fraction of pixels to shift at most in each direction.")
    parser.add_argument('--debug', action='store_true',
                        help="Save weights by TensorBoard")
    parser.add_argument('--save_dir', default='./result')
    parser.add_argument('-t', '--testing', action='store_true',
                        help="Test the trained model on testing dataset")
    parser.add_argument('--digit', default=5, type=int,
                        help="Digit to manipulate")
    parser.add_argument('-w', '--weights', default=None,
                        help="The path of the saved weights. Should be specified when testing")
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)



    (x_train, y_train), (x_test, y_test)= load_blood()
    model, eval_model= CapsNet(input_shape=x_train.shape[1:],
                               n_class=len(np.unique(np.argmax(y_train, 1))),
                               routings=args.routings)

    model.summary()
    model.save("result/model.h5")
    model.save_weights("result/weights")
    tf.keras.utils.plot_model(model, to_file="result/my_model.png", show_shapes=True)
    # train or test
    model= train(model=model, data=((x_train, y_train), (x_test, y_test)), args=args)

    train_history= save_plothistory(model=model)

    accuracy, prediction, accuracy1, auc = test(model=eval_model, data=(x_test, y_test), args=args)


    df=pd.DataFrame([accuracy])
    df.to_csv('result/accuracy.csv')

    print(accuracy)
    print(accuracy1)
    print(auc)