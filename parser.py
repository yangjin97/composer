from mido import Message, MidiFile, MidiTrack
import numpy as np
import time

NUM_NOTES = 89
MAX_TIME = 10000

class Parser():
  def __init__(self, midi_file):
    self.midi_file = MidiFile(midi_file)

  def __merge_tracks(self, tracks):
    times = []
    for track in tracks:
      times += list(track.keys())
    times = sorted(list(set(times)))

    for track in tracks:
      for i in range(len(times)):
        if times[i] not in track:
          track[times[i]] = track[times[i-1]]

    song = {}
    for time in times:
      if time not in song:
        song[time] = np.zeros(NUM_NOTES)
      for track in tracks:
        song[time] += track[time]

    for time in song:
      song[time] = np.clip(song[time], 0.0, 1.0)
    return song, times[-1]

  def parse(self):
    tracks = []
    min_time = MAX_TIME
    for i, track in enumerate(self.midi_file.tracks):
      mytrack = {}
      mytrack[0] = np.zeros(NUM_NOTES)

      time = 0
      prev_time = 0
      for msg in track:
        if msg.is_meta:
          continue

        prev_time = time
        time += msg.time
        if msg.time > 0:
          min_time = min(min_time, msg.time)

        if time not in mytrack:
          mytrack[time] = np.copy(mytrack[prev_time])

        if msg.type == 'note_on':
          if msg.velocity != 0:
            mytrack[time][msg.note] = 1.0
          else:
            mytrack[time][msg.note] = 0.0
        elif msg.type == 'note_off':
          mytrack[time][msg.note] = 0.0
      # for time in mytrack:
        # if time <= 5120:
          # print (time)
          # print (mytrack[time])
      # print ("-----------")
      tracks.append(mytrack)
    song, length = self.__merge_tracks(tracks)

    # for time in song:
      # if time <= 5120:
        # print (time)
        # print (song[time])
    
    # assume min_time divisible by length
    frames = []
    for time in range(0, length, min_time):
      if time in song:
        frames.append(song[time])
      else:
        frames.append(frames[-1])

    # print (self.midi_file.tracks)
    # print (frames[:20])
    
    return frames, len(self.midi_file.tracks), min_time

  def make_midi(self, frames, num_tracks, velocity, min_time):
    midi_output = MidiFile()
    for i in range(num_tracks):
      midi_output.tracks.append(MidiTrack())

    last_change = np.zeros(num_tracks, dtype=int)
    note_channel = np.full(NUM_NOTES, -1)

    for i in range(len(frames)):
      frame = frames[i]
      if i == 0:
        last_frame = np.zeros(NUM_NOTES)
      else:
        last_frame = frames[i-1]

      # only have 1 channel for now
      channel = 0
      for note in range(NUM_NOTES):
        # note pressed
        if frame[note] == 1 and last_frame[note] == 0:
          midi_output.tracks[channel].append(Message('note_on', note=note, velocity=velocity, time=min_time*i-last_change[channel]))
          note_channel[note] = channel
          last_change[channel] = min_time*i
        # note released
        elif frame[note] == 0 and last_frame[note] == 1:
          midi_output.tracks[note_channel[note]].append(Message('note_off', note=note, velocity=0, time=min_time*i-last_change[note_channel[note]]))
          last_change[note_channel[note]] = min_time*i

    # for track in midi_output.tracks:
      # for msg in track:
        # print (msg)
    midi_output.ticks_per_beat = self.midi_file.ticks_per_beat
    midi_output.save('output.mid')

# p = Parser('bach/chorales/01ausmei.mid')
# p = Parser('mozart_turkish_march.mid')
# p = Parser('mozart/mz_311_1.mid')
# p = Parser('chopin_nocturne_9_2.mid')
# frames, num_tracks, min_time = p.parse()
# p.make_midi(frames, num_tracks, 64, min_time)
# p.parse()
