from backend.utils import load_resources, extract_features, predict_class

from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
from PIL import ImageTk,Image

pca, filler_track, model = load_resources()

def browseFile():
    global filler_track, pca, model, v
    root.filename = filedialog.askopenfilename(initialdir="~", title="Select file",filetypes=(("wav files", "*.wav"), ("all files", "*.*")))
    s = root.filename
    X = extract_features(s, filler_track, pca)
    print('Loading...')
    p = predict_class(X, model)
    v.set(p)

root = Tk()
root.title("Genre guesser")
root.geometry("800x500")
root.config(background='#000000')

frame = Frame(root, bg='#000000')
left_frame = Frame(root, bg='#000000')
right_frame = Frame(root, bg='#000000')
frame2 = Frame(root, bg='#000000')

v = StringVar()
w = Label(root, text="Rouge", fg="red", textvariable=v, font=('monospace', 20))
w.pack()

bg_button = Button(frame2, text="Choose a song", bg='#cccccc', fg="black", command=browseFile)
bg_button.pack()

frame.pack(expand=YES)
frame2.pack(expand=YES, fill=Y)

root.mainloop()
