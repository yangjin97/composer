import tensorflow as tf

class LSTM():
  def __init__(self, time_steps, num_features):
    self.time_steps = time_steps
    self.num_features = num_features
    # self.optimizer = tf.train.RMSPropOptimizer()
    self.optimizer = tf.train.AdamOptimizer()
    # self.num_units = 128
    self.build_network()
    # self.build_loss()

  def build_network(self):
    with tf.variable_scope("global"):
      self.frames = []
      for i in range(self.time_steps):
        self.frames.append(tf.placeholder(tf.float32, [None, self.num_features])) 

      with tf.variable_scope('LSTM1'):
        self.lstm1 = tf.contrib.rnn.BasicLSTMCell(2*self.num_features)
        outputs, _ = tf.contrib.rnn.static_rnn(cell=self.lstm1, inputs=self.frames, dtype="float32")

      for i in range(len(outputs)):
        outputs[i] = tf.layers.dense(outputs[i], 2*self.num_features, activation=tf.sigmoid)

      with tf.variable_scope('LSTM2'):
        self.lstm2 = tf.contrib.rnn.BasicLSTMCell(self.num_features)
        outputs, _ = tf.contrib.rnn.static_rnn(cell=self.lstm2, inputs=outputs, dtype="float32")
      # state = self.init_state
      # print (len(outputs))
      # print (outputs[0].shape)
      # self.output = tf.layers.dense(outputs[0][-1][None,:], self.num_features, activation=tf.sigmoid)[0]
      self.output = outputs[-1]
      # print (self.output.shape)

      # self.loss = 0.0
      # for i in range(self.frames.shape[0]-1):
        # output, state = self.lstm(self.frames[i][None,:], state)
        # self.loss += tf.square(self.frames[i], output)
      # self.loss /= self.frames.shape[0]

      # self.loss = tf.losses.mean_squared_error(self.frames, tf.sigmoid(outputs[0]))
      self.target = tf.placeholder(tf.float32, [None, self.num_features])
      self.loss = tf.losses.mean_squared_error(self.target, self.output)

      tvars = tf.trainable_variables(scope="global")
      self.grads = tf.gradients(self.loss, tvars)

      self.train_op = self.optimizer.apply_gradients(zip(self.grads, tvars))
