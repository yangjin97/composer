from parser import Parser
from network import LSTM
import tensorflow as tf
import numpy as np

class Runner():
  def __init__(self):
    self.parser = Parser('bach/chorales/01ausmei.mid')
    self.frames, self.num_tracks, self.min_time = self.parser.parse()
    self.time_steps = len(self.frames)
    self.num_features = len(self.frames[0])
    self.network = LSTM(self.time_steps, self.num_features)

  def train(self):
    feed_dict = {
        self.network.frames: self.frames
        }

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      loss, _ = sess.run([self.network.loss, self.network.train_op], \
          feed_dict=feed_dict)
    # return loss

  def predict(self):
    feed_dict = {
        self.network.input: self.frames[0][None,:]
        }

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      outputs = sess.run(self.network.outputs, feed_dict=feed_dict)

    get_note = lambda x : 1 if x > 0.5 else 0
    vget_note = np.vectorize(get_note)
    for i in range(len(outputs)):
      outputs[i] = vget_note(outputs[i].flatten())
    
    return outputs

def main():
  runner = Runner()
  runner.train()
  outputs = runner.predict()
  runner.parser.make_midi(outputs, runner.num_tracks, 64, runner.min_time)

if __name__ == "__main__":
  main()
