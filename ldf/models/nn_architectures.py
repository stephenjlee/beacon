import sys, os
import tensorflow as tf

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

from tensorflow.keras.layers import \
    Dense, \
    concatenate, \
    AveragePooling2D, \
    Dropout, \
    Flatten, \
    Input
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras import regularizers
from tensorflow.keras.applications import ResNet50

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
sys.path.append(os.environ.get("PROJECT_ROOT"))

def get_regularizer(reg_mode, reg_val):
    if reg_mode == 'L2':
        return regularizers.l2(reg_val)
    elif reg_mode == 'L1':
        return regularizers.l1(reg_val)

def define_large_nn_model(input_dim, output_dim, do_val, reg_mode, reg_val):
    model = Sequential()
    model.add(Dense(20, input_dim=input_dim, kernel_initializer='normal', activation='relu',
                    kernel_regularizer=get_regularizer(reg_mode, reg_val)))
    model.add(Dropout(do_val))
    model.add(Dense(10, kernel_initializer='normal', activation='relu',
                    kernel_regularizer=get_regularizer(reg_mode, reg_val)))
    model.add(Dropout(do_val))
    model.add(Dense(output_dim, kernel_initializer='normal', activation='softplus',
                    kernel_regularizer=get_regularizer(reg_mode, reg_val)))
    return model

def define_shallow_nn_model(input_dim, output_dim, do_val, reg_mode, reg_val):
    model = Sequential()
    model.add(Dense(10, input_dim=input_dim, kernel_initializer='normal', activation='relu',
                    kernel_regularizer=get_regularizer(reg_mode, reg_val)))
    model.add(Dropout(do_val))
    model.add(
        Dense(5, kernel_initializer='normal', activation='relu', kernel_regularizer=get_regularizer(reg_mode, reg_val)))
    model.add(Dropout(do_val))
    model.add(Dense(output_dim, kernel_initializer='normal', activation='softplus',
                    kernel_regularizer=get_regularizer(reg_mode, reg_val)))
    return model


def define_xlarge_nn_model(input_dim, output_dim, do_val, reg_mode, reg_val):
    model = Sequential()
    model.add(Dense(20, input_dim=input_dim, kernel_initializer='normal', activation='relu',
                    kernel_regularizer=get_regularizer(reg_mode, reg_val)))
    model.add(Dropout(do_val))
    model.add(Dense(25, kernel_initializer='normal', activation='relu',
                    kernel_regularizer=get_regularizer(reg_mode, reg_val)))
    model.add(Dropout(do_val))
    model.add(Dense(30, kernel_initializer='normal', activation='relu',
                    kernel_regularizer=get_regularizer(reg_mode, reg_val)))
    model.add(Dropout(do_val))
    model.add(Dense(25, kernel_initializer='normal', activation='relu',
                    kernel_regularizer=get_regularizer(reg_mode, reg_val)))
    model.add(Dropout(do_val))
    model.add(Dense(output_dim, kernel_initializer='normal', activation='softplus',
                    kernel_regularizer=get_regularizer(reg_mode, reg_val)))
    return model

def get_larger_resnet(input_dim,
                      reg_mode,
                      reg_val,
                      do_val,
                      freeze_basemodel_layers=False):

    mlp = Sequential()
    mlp.add(Dense(20, input_dim=input_dim, kernel_initializer='normal', activation='relu',
                  kernel_regularizer=get_regularizer(reg_mode, reg_val)))  # Defining the first layer to be the number of features given
    mlp.add(Dropout(do_val))
    mlp.add(Dense(10, kernel_initializer='normal', activation='relu', kernel_regularizer=get_regularizer(reg_mode, reg_val)))
    mlp.add(Dropout(do_val))
    mlp.add(
        Dense(10, kernel_initializer='normal', activation='relu', kernel_regularizer=get_regularizer(reg_mode, reg_val)))

    try:
        base_model = ResNet50(weights="imagenet", include_top=False,
                         input_tensor=Input(shape=(224, 224, 3)))
    except:
        base_model = load_model(os.path.join(os.environ.get('PROJECT_DATA'), 'resnet50_weights.h5'))

    # construct the head of the model that will be placed on top of
    # the base model
    head_model = base_model.output
    head_model = AveragePooling2D(pool_size=(7, 7))(head_model)
    head_model = Flatten(name="flatten")(head_model)
    head_model = Dense(256, activation="relu")(head_model)
    head_model = Dropout(do_val)(head_model)
    head_model = Dense(10, activation="softmax")(head_model)

    # create the input to our final set of layers as the *output* of both
    # the MLP and CNN
    combined_input = concatenate([mlp.output, head_model])

    # our final FC layer head will have two dense layers, the final one
    # being our regression head
    x = Dense(10, activation="relu")(combined_input)
    x = Dense(2, activation="softplus")(x)

    # our final model will accept categorical/numerical data on the MLP
    # input and images on the CNN input, outputting a single value (the
    # predicted price of the house)
    model = Model(inputs=[mlp.input, base_model.input], outputs=x)

    # loop over all layers in the base model and freeze them so they will
    # *not* be updated during the training process
    if freeze_basemodel_layers:
        for layer in base_model.layers:
            layer.trainable = False

    return model
