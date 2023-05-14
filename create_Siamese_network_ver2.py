import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop

class CREATE_SIAMESE_NETWORK_MODEL:
    def create_network(self):
        input = Input(shape=(28, 28, 1), name='base_input')
        x = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1')(input)
        x = MaxPooling2D(pool_size=(2, 2), name='maxpool1')(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2')(x)
        x = MaxPooling2D(pool_size=(2, 2), name='maxpool2')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3')(x)
        x = MaxPooling2D(pool_size=(2, 2), name='maxpool3')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv4')(x)
        x = MaxPooling2D(pool_size=(2, 2), name='maxpool4')(x)
        x = Flatten(name='flatten_input')(x)
        x = Dense(128, activation='relu', name='first_base_dense')(x)
        x = Dropout(0.1, name='first_dropout')(x)
        x = Dense(128, activation='relu', name='second_base_dense')(x)
        x = Dropout(0.1, name='second_dropout')(x)
        x = Dense(128, activation='relu', name='third_base_dense')(x)
        
        return Model(inputs=input, outputs=x)
    
    def euclidean_distance(self, vects):
        x, y = vects
        sum_square = tf.math.reduce_sum(tf.math.square(x - y), axis=1, keepdims=True)
        return tf.math.sqrt(tf.math.maximum(sum_square, 1e-7))
    
    def contrastive_loss_with_margin(self, margin):
        def contrastive_loss(y_true, y_pred):
            square_pred = tf.math.square(y_pred)
            margin_square = tf.math.square(tf.math.maximum(margin - y_pred, 0))
            return tf.math.reduce_mean(y_true * square_pred + (1 - y_true) * margin_square)
        return contrastive_loss
    
    def create_model(self):
        base_model = self.create_network()
        input_a = Input(shape=(28, 28), name='left_input')
        input_b = Input(shape=(28, 28), name='right_input')

        reshape_a = Reshape((28, 28, 1))(input_a)
        reshape_b = Reshape((28, 28, 1))(input_b)

        vector_output_a = base_model(reshape_a)
        vector_output_b = base_model(reshape_b)

        output = Lambda(lambda x: self.euclidean_distance(x), name='output_layer')([vector_output_a, vector_output_b])

        model = Model([input_a, input_b], output)
        rms = RMSprop()
        model.compile(loss=self.contrastive_loss_with_margin(margin=1), optimizer=rms)
        
        return model