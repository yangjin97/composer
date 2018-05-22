import tensorflow as tf

class LSTM():
  def __init__(self, time_steps, num_features):
    self.time_steps = time_steps
    self.num_features = num_features
    self.optimizer = tf.train.RMSPropOptimizer(0.9)
    

  def build_network(self):
    with tf.variable_scope("global"):
      self.frames = tf.placeholder(tf.float32, [self.time_steps, self.num_features])
      self.lstm = tf.contrib.rnn.BasicLSTMCell(self.num_features)

      hidden_state = tf.zeros([1, lstm.state_size])
      current_state = tf.zeros([1, lstm.state_size])

      state = hidden_state, current_state

      self.loss = 0.0
      for i in range(len(self.frames)-1):
        output, state = self.lstm(self.frames[i], state)

        self.loss += tf.losses.mean_squared_error(self.frames[i+1])

      tvars = tf.trainable_variables(scope="global")
      grads = tf.gradients(self.loss, tvars)

      self.train_op = self.optimizer.apply_gradients(zip(grads, tvars))
