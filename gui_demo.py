from tkinter import *

class MultiheadSiameseNetGuiDemo:
  
  def __init__(self, master):
    frame = Frame(master)
    frame.pack()
    
    self.first_sentence_entry = Entry(frame, width=50)
    self.second_sentence_entry = Entry(frame, width=50)
    self.predictButton = Button(frame, text='Predict', command=self.predict)
    self.resultLabel = Label(frame, text='Result')
    
    self.first_sentence_entry.pack(side=LEFT, fill=X)
    self.second_sentence_entry.pack(side=LEFT, fill=X)
    self.predictButton.pack(side=LEFT)
    self.resultLabel.pack(side=BOTTOM)
    
    
  def predict(self):
    print('prediction')
  
  
root = Tk()
gui = MultiheadSiameseNetGuiDemo(root)
root.mainloop()