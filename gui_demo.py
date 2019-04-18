import os
from tkinter import *
from tkinter import messagebox

import tensorflow as tf

from models.model_type import MODELS
from utils.data_utils import DatasetVectorizer
from utils.other_utils import init_config
from utils.other_utils import logger


class MultiheadSiameseNetGuiDemo:
  
  def __init__(self, master):
    frame = Frame(master)
    frame.pack()
    
    self.first_sentence_entry = Entry(frame, width=50)
    self.second_sentence_entry = Entry(frame, width=50)
    self.predictButton = Button(frame, text='Predict', command=self.predict)
    self.clearButton = Button(frame, text='Clear', command=self.clear)
    self.resultLabel = Label(frame, text='Result')
    
    self.main_config = init_config()
    self.model_dir = str(self.main_config['DATA']['model_dir'])
    
    model_dirs = [os.path.basename(x[0]) for x in os.walk(self.model_dir)]
    
    variable = StringVar(master)
    variable.set('Choose a model...')
    self.model_type = OptionMenu(master, variable, *model_dirs, command=self.load_model)
    
    self.first_sentence_entry.pack(side=LEFT, fill=X)
    self.second_sentence_entry.pack(side=LEFT, fill=X)
    self.predictButton.pack(side=LEFT)
    self.clearButton.pack(side=LEFT)
    self.resultLabel.pack(side=BOTTOM)
    self.model_type.pack(side=LEFT)
    
    self.vectorizer = DatasetVectorizer(self.model_dir)
    
    self.max_doc_len = self.vectorizer.max_sentence_len
    self.vocabulary_size = self.vectorizer.vocabulary_size
    
    self.session = tf.Session()
    self.model = None
  
  def predict(self):
    if self.model:
      sentence1 = self.first_sentence_entry.get()
      sentence2 = self.second_sentence_entry.get()
      x1_sen = self.vectorizer.vectorize(sentence1)
      x2_sen = self.vectorizer.vectorize(sentence2)
      feed_dict = {self.model.x1: x1_sen, self.model.x2: x2_sen, self.model.is_training: False}
      prediction = self.session.run([self.model.predictions], feed_dict=feed_dict)
      self.resultLabel['text'] = prediction
    else:
      messagebox.showerror("Error", "Choose a model to make a prediction.")
  
  def clear(self):
    self.first_sentence_entry.delete(0, 'end')
    self.second_sentence_entry.delete(0, 'end')
    self.resultLabel['text'] = ''
  
  def load_model(self, model_name):
    logger.info('Loading model: %s', model_name)
    
    model = MODELS[model_name.split('_')[0]]
    model_config = init_config(model_name.split('_')[0])
    
    self.model = model(self.max_doc_len, self.vocabulary_size, self.main_config, model_config)
    saver = tf.train.Saver()
    last_checkpoint = tf.train.latest_checkpoint('{}/{}'.format(self.model_dir, model_name))
    saver.restore(self.session, last_checkpoint)
    logger.info('Loaded model from: %s', last_checkpoint)


root = Tk()
gui = MultiheadSiameseNetGuiDemo(root)
root.mainloop()
