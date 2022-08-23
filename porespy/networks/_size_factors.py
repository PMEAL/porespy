import numpy as np
import scipy.ndimage as spim
import math
from porespy import settings
from porespy.tools import extend_slice
from scipy.ndimage import zoom as zm
from skimage.morphology import ball
from porespy.tools import get_tqdm
tqdm = get_tqdm()


def diffusive_size_factor_AI(regions, throat_conns, model,
                             g_train, voxel_size=1):
    '''
    Parameters
    ----------
    regions : ndarray
        A segmented 3D image of pore regions/a pair of two regions.
    throat_conns : array
        An Nt by 2 array containing the throat connections. The indices orders in
        throat_conns start from 0 to be consistent with network extraction method.
    model : tensorflow model
        The trained model to be used for prediction.
    g_train : array
        The training data distribution. This will be used for denormalizing
        the prediction.
    voxel_size : scalar, optional
        Voxel size of the image. The default is 1.

    Returns
    -------
    diff_size_factor : array
        An array of length conns containing diffusive size factor of the conduits
        in the segmented image (regions).

    Examples
    --------
    `Click here
    <https://porespy.org/examples/networks/reference/diffusive_size_factor_AI.html>`_
    to view online example.

    """
    '''
    import tensorflow as tf
    if g_train is None:
        raise ValueError("Training ground truth data must be given\
                         to be used for normalizing the test data")
    its = -1
    pairs = np.empty((len(throat_conns), 64, 64, 64), dtype='int32')
    diff_size_factor = []
    zm_ratios = []
    desc = 'Preparing images tensor'
    for i in tqdm(np.arange(len(throat_conns)), desc=desc, **settings.tqdm):
        cn = throat_conns[i]
        # crop two pore regions and label them as 1,2
        bb = _calc_bound_box_bi(regions, cn[0], cn[1])
        roi_crop = np.copy(regions[bb[0]:bb[3], bb[1]:bb[4], bb[2]:bb[5]])
        # label two pore regions as 1,2
        roi_masked = _create_labeled_pair(cn, roi_crop)
        # resize to AI input size
        resize_result = _resize_to_AI_input(roi_masked)
        roi_resized = resize_result['resized_im']
        zm_ratio = resize_result['zm_ratio']
        zm_ratios.append(zm_ratio)
        its = its+1
        pairs[its, :, :, :] = roi_resized.copy()
    test_data = _convert_to_tf_image_datatype(pairs, n_image=len(throat_conns))
    if len(throat_conns) < 16:
        batch_size = 1
    else:
        batch_size = 16
    test_steps = math.ceil(len(throat_conns) / batch_size)
    predictions = model.predict(test_data, steps=test_steps)
    predictions = np.squeeze(predictions)
    denorm_size_factor = _denorm_predict(predictions, g_train)
    g = denorm_size_factor * voxel_size * (1/np.array(zm_ratios))
    diff_size_factor = g
    if len(throat_conns) > 1:
        tf.keras.backend.clear_session()
    return diff_size_factor


def _calc_bound_box_bi(regions, pore_1, pore_2):
    '''

    Parameters
    ----------
    regions : ndarray
        3D image of the segmented regions from which the bounding box of
        a local pore to pore region will be extracted.
    pore_1 : scalar
        Label of pore region1 in a pair of pore to pore connection (throat)
    pore_2 : scalar
        Label of pore region1 in a pair of pore to pore connection (throat)

    Returns
    -------
    b_box : array
        An array containing the coordinates of the bounding box that covers
        region pore_1 and region pore_2. The orders of coordinates are:
        x_min, y_min, z_min, x_max, y_max, z_max.

    '''
    slice_x, slice_y, slice_z = spim.find_objects(regions == pore_1+1)[0]
    slice_x2, slice_y2, slice_z2 = spim.find_objects(regions == pore_2+1)[0]
    min_box = [min(slice_x.start, slice_x2.start),
               min(slice_y.start, slice_y2.start),
               min(slice_z.start, slice_z2.start)]
    max_box = [max(slice_x.stop, slice_x2.stop), max(slice_y.stop, slice_y2.stop),
               max(slice_z.stop, slice_z2.stop)]
    b_box = np.hstack([min_box, max_box])
    return b_box


def _resize_to_AI_input(im):
    '''
    Parameters
    ----------
    im : ndarray
        3D image of a pair of pore to pore regions (conduit) cropped from
        segmented image of a porous medium.

    Returns
    -------
    result : dict
        Contains Resized image of the cropped regions and its zoom ratio.
        If original image(im) is of shape (n,n,n) resizing includes one step
        of zoom to a (64,64,64) image. Otherwise, the original image
        will be first zero padded to its maximum dimension (nmax,nmax,nmax)
        before zoom step.

    '''
    if len(np.unique(im.shape)) != 1:
        x, y, z = np.shape(im)
        max_size = max([x, y, z])
        x_diff, y_diff, z_diff = max_size-np.array([x, y, z])
        XB = int(np.round(x_diff/2))  # padding before axis x
        XA = int(x_diff-XB)  # padding after axis x
        YB = int(np.round(y_diff/2))
        YA = int(y_diff-YB)
        ZB = int(np.round(z_diff/2))
        ZA = int(z_diff-ZB)
        im = np.pad(im, ((XB, XA), (YB, YA), (ZB, ZA)), 'constant',
                    constant_values=0)
    zm_ratio = 64/im.shape[0]
    resized_im = zm(im, zoom=[zm_ratio, zm_ratio, zm_ratio], order=0)
    result = {'zm_ratio': zm_ratio, 'resized_im': resized_im}
    return result


def find_conns(im):
    '''
    Parameters
    ----------
    im : ndarray
        A segmented image of a porous medium.

    Returns
    -------
    t_conns : array
        An Nt by 2 addat containing the throats' connections in the segmented image.

    '''
    struc_elem = ball
    slices = spim.find_objects(im)
    Ps = np.arange(1, np.amax(im)+1)
    t_conns = []
    d = 'Finding neighbouring regions'
    for i in tqdm(Ps, desc=d, **settings.tqdm):
        pore = i - 1
        if slices[pore] is None:
            continue
        s = extend_slice(slices[pore], im.shape)
        sub_im = im[s]
        pore_im = sub_im == i
        im_w_throats = spim.binary_dilation(input=pore_im, structure=struc_elem(1))
        im_w_throats = im_w_throats*sub_im
        Pn = np.unique(im_w_throats)[1:] - 1
        for j in Pn:
            if j > pore:
                t_conns.append([pore, j])
    return t_conns


def _create_labeled_pair(cn, im):
    '''
    Parameters
    ----------
    cn : array
        An array containing two elements [region1,region2] that are labels of
        pore regions in a pair (im).
    im : ndarray
        3D image of a pair of neighbouring pores.

    Returns
    -------
    roi_masked : ndarray
        labeled image of two pore regions (im) where solid, region1, and region2 are
        labeled as 0, 1, and 2, respectively.

    '''
    # create labeled image for AI : labels (0,1,2) for (solid,pore1,pore2)
    roi_crop = np.copy(im)
    mask1 = np.array(np.where(roi_crop == cn[0]+1, roi_crop, 0),
                     dtype=bool)
    mask2 = np.array(np.where(roi_crop == cn[1]+1, roi_crop, 0),
                     dtype=bool)
    roi_masked = 1*mask1 + 2*mask2
    return roi_masked


def _convert_to_tf_image_datatype(pair, n_image):
    '''

    Parameters
    ----------
    pair : ndarray
        Labeled pore to pore regions.
    n_image : scalar
        Number of pairs of conduit regions.

    Returns
    -------
    test_data : tensorflow data
        Tensorflow data type test data. The test data may include 1 or more pairs
        of conduit images.

    '''
    import tensorflow as tf
    # create a tensor of size (n_image, pair_shape)
    data_ims = np.zeros(shape=(n_image, 64, 64, 64, 1))
    data = np.expand_dims(pair, axis=-1)
    if n_image == 1:
        data_ims[0, :, :, :] = data
        BS = 1
    else:
        data_ims = data
        if n_image < 16:
            BS = 1
        else:
            BS = 16
    # create test_data as a tensorflow dataset type
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    test_data = tf.data.Dataset.from_tensor_slices(data_ims)
    test_data = test_data.cache()
    test_data = test_data.batch(BS)   # batch size = 1
    test_data = test_data.prefetch(AUTOTUNE)
    return test_data


def _denorm_predict(prediction, g_train):
    '''


    Parameters
    ----------
    prediction : array
        Normalized predicted diffusive size factors for conduit images.
    g_train : array
        Diffusive size factors used in training AI.

    Returns
    -------
    denorm : array
        Denormalized predicted diffusive size factor for conduit images.

    '''
    from sklearn import preprocessing
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    train_N = scaler.fit_transform(g_train.reshape(-1, 1))
    denorm = scaler.inverse_transform(X=prediction.reshape(-1, 1))
    denorm = np.squeeze(denorm)
    return denorm


def _id_block(x, filters, kernel_size):
    from tensorflow.keras import layers as ly
    from tensorflow.keras.initializers import glorot_uniform
    f1, f2, f3 = filters
    k = kernel_size
    x_orig = x

    x = ly.Conv3D(filters=f1, kernel_size=(1, 1, 1),
                  strides=(1, 1, 1), padding='valid',
                  kernel_initializer=glorot_uniform(seed=0))(x)
    x = ly.BatchNormalization(axis=-1)(x)
    x = ly.Activation('relu')(x)

    x = ly.Conv3D(filters=f2, kernel_size=(k, k, k),
                  strides=(1, 1, 1), padding='same',
                  kernel_initializer=glorot_uniform(seed=0))(x)
    x = ly.BatchNormalization(axis=-1)(x)
    x = ly.Activation('relu')(x)

    x = ly.Conv3D(filters=f3, kernel_size=(1, 1, 1),
                  strides=(1, 1, 1), padding='valid',
                  kernel_initializer=glorot_uniform(seed=0))(x)
    x = ly.BatchNormalization(axis=-1)(x)

    x = ly.Add()([x, x_orig])
    x = ly.Activation('relu')(x)

    return x


def _conv_block(x, filters, kernel_size, stride):
    from tensorflow.keras import layers as ly
    from tensorflow.keras.initializers import glorot_uniform
    f1, f2, f3 = filters
    k = kernel_size
    s = stride
    x_orig = x

    x = ly.Conv3D(filters=f1, kernel_size=(1, 1, 1),
                  strides=(s, s, s), padding='valid',
                  kernel_initializer=glorot_uniform(seed=0))(x)
    x = ly.BatchNormalization(axis=-1)(x)
    x = ly.Activation('relu')(x)

    x = ly.Conv3D(filters=f2, kernel_size=(k, k, k),
                  strides=(1, 1, 1), padding='same',
                  kernel_initializer=glorot_uniform(seed=0))(x)
    x = ly.BatchNormalization(axis=-1)(x)
    x = ly.Activation('relu')(x)

    x = ly.Conv3D(filters=f3, kernel_size=(1, 1, 1),
                  strides=(1, 1, 1), padding='valid',
                  kernel_initializer=glorot_uniform(seed=0))(x)
    x = ly.BatchNormalization(axis=-1)(x)

    x_shortcut = ly.Conv3D(filters=f3, kernel_size=(1, 1, 1),
                           strides=(s, s, s), padding='valid',
                           kernel_initializer=glorot_uniform(seed=0))(x_orig)
    x_shortcut = ly.BatchNormalization(axis=-1)(x_shortcut)

    x = ly.Add()([x, x_shortcut])
    x = ly.Activation('relu')(x)

    return x


def _resnet3d(input_shape=(64, 64, 64, 1)):
    from tensorflow.keras import layers as ly
    from tensorflow.keras.initializers import glorot_uniform
    from tensorflow.keras.models import Model
    x_in = ly.Input(shape=input_shape)

    x = ly.ZeroPadding3D(padding=(3, 3, 3))(x_in)

    # stage 1
    x = ly.Conv3D(filters=64, kernel_size=(7, 7, 7), strides=(2, 2, 2),
                  kernel_initializer=glorot_uniform(seed=0))(x)
    x = ly.BatchNormalization(axis=-1)(x)
    x = ly.Activation('relu')(x)
    x = ly.MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2))(x)

    # stage 2
    x = _conv_block(x, filters=(64, 64, 256), kernel_size=3, stride=1)
    x = _id_block(x, filters=(64, 64, 256), kernel_size=3)
    x = _id_block(x, filters=(64, 64, 256), kernel_size=3)

    # stage 3
    x = _conv_block(x, filters=(128, 128, 512), kernel_size=3, stride=2)
    x = _id_block(x, filters=(128, 128, 512), kernel_size=3)
    x = _id_block(x, filters=(128, 128, 512), kernel_size=3)
    x = _id_block(x, filters=(128, 128, 512), kernel_size=3)

    # stage 4
    x = _conv_block(x, filters=(256, 256, 1024), kernel_size=3, stride=2)
    x = _id_block(x, filters=(256, 256, 1024), kernel_size=3)
    x = _id_block(x, filters=(256, 256, 1024), kernel_size=3)
    x = _id_block(x, filters=(256, 256, 1024), kernel_size=3)
    x = _id_block(x, filters=(256, 256, 1024), kernel_size=3)
    x = _id_block(x, filters=(256, 256, 1024), kernel_size=3)

    # stage 5
    x = _conv_block(x, filters=(512, 512, 2048), kernel_size=3, stride=2)
    x = _id_block(x, filters=(512, 512, 2048), kernel_size=3)
    x = _id_block(x, filters=(512, 512, 2048), kernel_size=3)

    # average pooling
    x = ly.AveragePooling3D(pool_size=(2, 2, 2))(x)

    # output layer
    x = ly.Flatten()(x)

    x = ly.Dense(units=512, kernel_initializer=glorot_uniform(seed=0))(x)
    x = ly.Activation(activation='relu')(x)
    x = ly.Dropout(rate=0.5)(x)

    x = ly.Dense(units=1, kernel_initializer=glorot_uniform(seed=0))(x)
    x = ly.Activation(activation='linear')(x)

    # create model
    model = Model(inputs=x_in, outputs=x, name='resnet3d')

    return model


def create_model():
    '''
    Returns
    -------
    model : tensorflow model
        ResNet50 model built using convolutional and identity blocks.

    '''
    from tensorflow.keras.optimizers import Adam
    model = _resnet3d()
    model.compile(loss='mse', optimizer=Adam(lr=1e-4), metrics=['mse'])
    return model
