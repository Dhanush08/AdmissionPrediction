from tkinter import *
from tkinter import ttk
from joblib import load
import numpy as np

def makeform(root, fields):
    entries = {}
    for field in fields:
        row = Frame(root)
        lab = Label(row, width=22, text=field+": ", anchor='w')
        ent = Entry(row)
        ent.insert(0, "0")
        row.pack(side=TOP, fill=X, padx=5, pady=5)
        lab.pack(side=LEFT)
        ent.pack(side=RIGHT, expand=YES, fill=X)
        entries[field] = ent
    return entries

Normaliser = load('normalise.bin')
LinearReg = load('linearreg.bin')
fields = ("GRE Score (0-340)", "TOEFL Score (0-120)", "University Rating (1-5)",
          "SOP Score (0-5)", "LOR Score (0-5)", "CGPA (0-10)", "Researches Done (0/1)")
window = Tk()
window.title("Predicting Graduate Admissions Using ML")
Label(window, text="Enter your scores below :")

ents = makeform(window, fields)
window.geometry('600x600')

def popup_bonus(val):
    win = Toplevel()
    win.wm_title("Result")
    r = round(float(val.item()*100), 2)
    if(r >= 100):
       r = 100
    if(r<0):
       r=0
    l = Label(
        win, text="Your chance of admission in the given university is {} %".format(r))
    l.grid(row=0, column=0)
    b = ttk.Button(win, text="Okay", command=win.destroy)
    b.grid(row=1, column=0)

def clicked(e):
    sc = []
    for i in fields:
        sc.append(e[i].get())
    x = np.array([sc, sc])
    x = Normaliser.transform(x)
    y = LinearReg.predict(x)
    popup_bonus(y[0])

b1 = Button(window, text='Submit',
            command=(lambda e=ents: clicked(e)))
b1.pack(side=LEFT, padx=5, pady=5)
window.mainloop()
