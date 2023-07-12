# import os.path
#
# import keras
# import tensorflow as tf
# from keras import backend as K
#
#
# def dice_coefficient(y_true, y_pred, smooth=1):
#     """ Dice coefficient metric function """
#     indices = K.argmax(y_pred, 3)
#     indices = K.reshape(indices, [-1, 224, 224, 1])
#
#     true_cast = y_true
#     indices_cast = K.cast(indices, dtype='float32')
#
#     axis = [1, 2, 3]
#     intersection = K.sum(true_cast * indices_cast, axis=axis)
#     union = K.sum(true_cast, axis=axis) + K.sum(indices_cast, axis=axis)
#     dice = K.mean((2. * intersection + smooth) / (union + smooth), axis=0)
#
#     return dice
#
#
# def load_model(model_name: str) -> keras.Model:
#     """ Loads and returns the loaded model """
#     path_to_model = os.path.join(os.getcwd(), "assets/model", model_name)
#     model = tf.keras.models.load_model(
#         path_to_model,
#         custom_objects={
#             "dice_coef": dice_coefficient,
#         }
#     )
#     return model
