import os
from tkinter import *
from tkinter import messagebox

import numpy as np
import tensorflow as tf

from models.model_type import MODELS
from utils import visualization
from utils.data_utils import DatasetVectorizer
from utils.other_utils import init_config
from utils.other_utils import logger

GUI_FONT_SIZE = 14
SAMPLE_SENTENCE1 = 'Wet brown dog swims towards camera.'
SAMPLE_SENTENCE2 = 'A dog is in the water.'


class MultiheadSiameseNetGuiDemo:
    
    def __init__(self, master):
        self.frame = master
        self.frame.title('Multihead Siamese Nets')
        
        sample1 = StringVar(master, value=SAMPLE_SENTENCE1)
        sample2 = StringVar(master, value=SAMPLE_SENTENCE2)
        self.first_sentence_entry = Entry(self.frame, width=50,
                                          font="Helvetica {}".format(GUI_FONT_SIZE),
                                          textvariable=sample1)
        self.second_sentence_entry = Entry(self.frame, width=50,
                                           font="Helvetica {}".format(GUI_FONT_SIZE),
                                           textvariable=sample2)
        self.predictButton = Button(self.frame, text='Predict',
                                    font="Helvetica {}".format(GUI_FONT_SIZE),
                                    command=self.predict)
        self.clearButton = Button(self.frame, text='Clear', command=self.clear,
                                  font="Helvetica {}".format(GUI_FONT_SIZE))
        self.resultLabel = Label(self.frame, text='Result',
                                 font="Helvetica {}".format(GUI_FONT_SIZE))
        self.first_sentence_label = Label(self.frame, text='Sentence 1',
                                          font="Helvetica {}".format(GUI_FONT_SIZE))
        self.second_sentence_label = Label(self.frame, text='Sentence 2',
                                           font="Helvetica {}".format(GUI_FONT_SIZE))
        
        self.main_config = init_config()
        self.model_dir = str(self.main_config['DATA']['model_dir'])
        
        model_dirs = [os.path.basename(x[0]) for x in os.walk(self.model_dir)]
        
        self.visualize_attentions = IntVar()
        self.visualize_attentions_checkbox = Checkbutton(master, text="Visualize attention weights",
                                                         font="Helvetica {}".format(
                                                             int(GUI_FONT_SIZE / 2)),
                                                         variable=self.visualize_attentions,
                                                         onvalue=1, offvalue=0)
        
        variable = StringVar(master)
        variable.set('Choose a model...')
        self.model_type = OptionMenu(master, variable, *model_dirs, command=self.load_model)
        self.model_type.configure(font=('Helvetica', GUI_FONT_SIZE))
        
        self.first_sentence_entry.grid(row=0, column=1, columnspan=4)
        self.first_sentence_label.grid(row=0, column=0, sticky=E)
        self.second_sentence_entry.grid(row=1, column=1, columnspan=4)
        self.second_sentence_label.grid(row=1, column=0, sticky=E)
        self.model_type.grid(row=2, column=1, sticky=W + E, ipady=1)
        self.predictButton.grid(row=2, column=2, sticky=W + E, ipady=1)
        self.clearButton.grid(row=2, column=3, sticky=W + E, ipady=1)
        self.resultLabel.grid(row=2, column=4, sticky=W + E, ipady=1)
        
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
            feed_dict = {self.model.x1: x1_sen, self.model.x2: x2_sen,
                         self.model.is_training: False}
            
            if self.visualize_attentions.get():
                prediction, at1, at2 = np.squeeze(
                    self.session.run(
                        [self.model.predictions, self.model.debug_vars['attentions_x1'],
                         self.model.debug_vars['attentions_x2']], feed_dict=feed_dict))
                visualization.visualize_attention_weights(at1, sentence1)
                visualization.visualize_attention_weights(at2, sentence2)
            else:
                prediction = np.squeeze(
                    self.session.run(self.model.predictions, feed_dict=feed_dict))
            
            prediction = np.round(prediction, 2)
            self.resultLabel['text'] = prediction
            if prediction < 0.5:
                self.resultLabel.configure(foreground="red")
            else:
                self.resultLabel.configure(foreground="green")
        else:
            messagebox.showerror("Error", "Choose a model to make a prediction.")
    
    def clear(self):
        self.first_sentence_entry.delete(0, 'end')
        self.second_sentence_entry.delete(0, 'end')
        self.resultLabel['text'] = ''
    
    def load_model(self, model_name):
        if 'multihead' in model_name:
            self.visualize_attentions_checkbox.grid(row=2, column=0, sticky=W + E, ipady=1)
        else:
            self.visualize_attentions_checkbox.grid_forget()
        tf.reset_default_graph()
        self.session = tf.Session()
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
