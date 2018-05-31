import tensorflow as tf

input_data = tf.placeholder(tf.float32, [10])
initial_state = tf.Variable(tf.truncated_normal([10]))
final_state = initial_state + input_data
num_steps = 10
session = tf.Session()
session.run(tf.global_variables_initializer())
x = session.run(tf.zeros_like(input_data))
state = session.run(initial_state)
for step in range(num_steps):
    state = session.run(final_state,
                        {input_data: x,
                         initial_state: state})
    x = state
