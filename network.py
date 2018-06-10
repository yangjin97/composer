import tensorflow as tf

class LSTM():
  def __init__(self, time_steps, num_features):
    self.time_steps = time_steps
    self.num_features = num_features
    self.optimizer = tf.train.RMSPropOptimizer(0.9)
    self.build_network()
    self.build_loss()

  def build_network(self):
    with tf.variable_scope("global"):
      self.input = tf.placeholder(tf.float32, [1, self.num_features])
      self.lstm = tf.contrib.rnn.BasicLSTMCell(self.num_features)

      hidden_state = tf.zeros([1, self.num_features])
      current_state = tf.zeros([1, self.num_features])
      self.init_state = hidden_state, current_state

      state = self.init_state
      self.outputs = []
      for i in range(self.time_steps):
        output, state = self.lstm(self.input, state)
        self.outputs.append(output)

  def build_loss(self):
    with tf.variable_scope("global"):
      self.frames = tf.placeholder(tf.float32, [self.time_steps, self.num_features])

      state = self.init_state
      self.loss = 0.0
      for i in range(self.frames.shape[0]-1):
        output, state = self.lstm(self.frames[i][None,:], state)

        self.loss += tf.losses.mean_squared_error(self.frames[i+1][None,:],
            output)

      tvars = tf.trainable_variables(scope="global")
      grads = tf.gradients(self.loss, tvars)

      self.train_op = self.optimizer.apply_gradients(zip(grads, tvars))
