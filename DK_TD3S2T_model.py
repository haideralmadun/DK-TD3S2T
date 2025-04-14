import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tcn import TCN
from custom_flood_loss import custom_flood_loss




class SpatialAttention(layers.Layer):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv1 = layers.Conv1D(1, kernel_size=1, activation='sigmoid', padding='same')

    def call(self, inputs):
        attention_outputs = []
        for input_tensor in inputs:
            attention_weights = self.conv1(input_tensor)
            attention_weights = tf.nn.softmax(attention_weights, axis=-1)
            attention_outputs.append(input_tensor * attention_weights)
        return tf.concat(attention_outputs, axis=-1)

class TemporalAttention(layers.Layer):
    def __init__(self):
        super(TemporalAttention, self).__init__()
        self.dense1 = layers.Dense(128, activation='relu')
        self.dense2 = layers.Dense(1)

    def call(self, inputs):
        temporal_features = self.dense1(inputs)
        attention_weights = self.dense2(temporal_features)
        attention_weights = tf.nn.softmax(attention_weights, axis=-2)
        attention_weights = tf.squeeze(attention_weights, axis=-1)
        attention_weights = tf.expand_dims(attention_weights, axis=-1)
        return tf.reduce_sum(inputs * attention_weights, axis=-2)




def DK_TD3S2T_model(n_steps_in, n_steps_out, n_features, n_seq, n_steps):
    input_shape = (n_steps_in, n_features)
    input_shape1 = (n_seq, n_steps, n_features)

    inputs = Input(shape=input_shape)
    inputs1 = Input(shape=input_shape1)

    # Left stream: TD-CNN + LSTM
    conv = layers.TimeDistributed(layers.Conv1D(64, 1, activation='relu'))(inputs1)
    conv = layers.TimeDistributed(layers.Conv1D(64, 1, activation='relu'))(conv)
    conv = layers.TimeDistributed(layers.MaxPooling1D(1))(conv)
    conv = layers.TimeDistributed(layers.Flatten())(conv)

    lstm = layers.LSTM(32, activation='relu', return_sequences=True)(conv)
    lstm = layers.LSTM(32, activation='relu', return_sequences=True)(lstm)
    lstm = TemporalAttention()(lstm)

    # Right stream: Spatial + TCN
    spatial = SpatialAttention()([inputs, inputs, inputs])
    tcn = layers.Conv1D(64, 1, activation='relu', padding='causal')(spatial)
    tcn = TCN(32, kernel_size=1, dilations=[1, 2, 4, 8], return_sequences=True)(tcn)
    tcn = TCN(32, kernel_size=1, dilations=[1, 2, 4, 8], return_sequences=True)(tcn)
    tcn = TemporalAttention()(tcn)

    # Merge and predict
    concat = layers.Concatenate(axis=-1)([lstm, tcn])
    outputs = layers.Dense(n_steps_out)(concat)

    model = Model(inputs=[inputs, inputs1], outputs=outputs)
    model.compile(optimizer='Nadam', loss=custom_flood_loss)
    return model
