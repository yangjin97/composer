from parser import Parser
from network import LSTM
import tensorflow as tf
import numpy as np

class Runner():
  def __init__(self, train_steps):
    self.parser = Parser('bach/chorales/01ausmei.mid')
    self.frames, self.num_tracks, self.min_time = self.parser.parse()
    self.train_steps = train_steps
    self.time_steps = len(self.frames)
    self.num_features = len(self.frames[0])
    self.network = LSTM(self.train_steps-1, self.num_features)

  def train(self, num_epoch, sess):
    for e in range(num_epoch):
      # batch_frames = []
      # batch_target = []
      feed_dict = {
          # self.network.frames: batch_frames,
          # self.network.target: batch_target
          }
      for step in range(self.train_steps-1):
        feed_dict[self.network.frames[step]] = \
            self.frames[step:step+self.time_steps-self.train_steps]
      feed_dict[self.network.target] = self.frames[self.train_steps:]
      loss, _, grads = sess.run([self.network.loss, self.network.train_op, \
              self.network.grads], feed_dict=feed_dict)
      if e % 10 == 0:
        print (e, loss)
      if loss < 0.0002:
        break

  def predict(self, sess):
    outputs = self.frames[:self.train_steps]
    for step in range(self.time_steps-self.train_steps):
      feed_dict = {}
      for small_step in range(self.train_steps-1):
        feed_dict[self.network.frames[small_step]] = \
            [outputs[-self.train_steps+1+small_step]]
      output = sess.run(self.network.output, feed_dict=feed_dict)
      outputs.append(output[0])

    get_note = lambda x : 1 if x > 0.9 else 0
    vget_note = np.vectorize(get_note)
    for i in range(len(outputs)):
      outputs[i] = vget_note(outputs[i].flatten())
    outputs.append(np.zeros(self.num_features))
    
    # print (outputs[-10:])
    return outputs

def main():
  runner = Runner(40)
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    runner.train(100000, sess)
    outputs = runner.predict(sess)
  # runner.parser.make_midi(runner.frames, runner.num_tracks, 64, runner.min_time)
  runner.parser.make_midi(outputs, runner.num_tracks, 64, runner.min_time)

if __name__ == "__main__":
  main()
