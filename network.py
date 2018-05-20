import tensorflow as tf

class LSTM():
  def __init__(self, time_steps, num_features):
    self.time_steps = time_steps
    self.num_features = num_features
    

  def build_network(self):
    self.frames = tf.placeholder(tf.float32, [time_steps, num_features])

    hidden_state = tf.zeros([1, lstm.state_size])
		current_state = tf.zeros([1, lstm.state_size])

    state = hidden_state, current_state

    for frame in self.frames:
      
