# # neural_network.py

# import tensorflow as tf

# def predict_ui_adjustments(data):
#     # Example neural network with dummy layers
#     model = tf.keras.Sequential([
#         tf.keras.layers.Dense(10, activation='relu'),
#         tf.keras.layers.Dense(5, activation='relu'),
#         tf.keras.layers.Dense(3, activation='softmax')
#     ])

#     # Convert the data to a tensor and reshape to make it 2D
#     input_data = tf.convert_to_tensor(data, dtype=tf.float32)

#     # Ensure that the input data has a batch dimension
#     input_data = tf.expand_dims(input_data, axis=0)  # Adds a batch dimension

#     prediction = model.predict(input_data)
#     return prediction

# neural_network.py

import tensorflow as tf

# Define the model globally (only once)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(your_input_dim_here,)),  # specify input shape
    tf.keras.layers.Dense(5, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

# You can compile the model if you plan to train it, otherwise skip compile()
# model.compile(optimizer='adam', loss='categorical_crossentropy')

def predict_ui_adjustments(data):
    input_data = tf.convert_to_tensor(data, dtype=tf.float32)
    input_data = tf.expand_dims(input_data, axis=0)
    prediction = model.predict(input_data)
    return prediction
