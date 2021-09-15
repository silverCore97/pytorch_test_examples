import sqlite3
import sys, tkinter
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

import numpy as np


# Erzeugt neues Fenster mit Schaltfläche Ende
def fenster():
    global neu
    neu = tkinter.Toplevel(main)
    fig = Figure(figsize=(5, 4), dpi=100)
    n=e1.get()
    m=e2.get()
    sql="SELECT epoch, accuracy, time FROM Data WHERE n = "+str(n)+" AND m = "+str(m)
    cursor.execute(sql)
    
    ep=[]
    ac=[]
    ti=[]
    for dsatz in cursor:
        ep.append(dsatz[0])
        ac.append(dsatz[1])
        ti.append(dsatz[2])
    fig.add_subplot().plot(ep, ac,label="accuracy for "+str(n)+" "+str(m))
    #fig.add_subplot().plot(ep, ti,label="timefor "+str(n)+" "+str(m))
    fig.legend()
    
    canvas = FigureCanvasTkAgg(fig, master=neu)  # A tk.DrawingArea.
    canvas.draw()
        
    # pack_toolbar=False will make it easier to use a layout manager later on.
    toolbar = NavigationToolbar2Tk(canvas, neu, pack_toolbar=False)
    toolbar.update()


    canvas.mpl_connect(
        "key_press_event", lambda event: print(f"you pressed {event.key}"))
    canvas.mpl_connect("key_press_event", key_press_handler)

    button = tkinter.Button(master=neu, text="Quit", command=neu.quit)

    # Packing order is important. Widgets are processed sequentially and if there
    # is no space left, because the window is too small, they are not displayed.
    # The canvas is rather flexible in its size, so we pack it last which makes
    # sure the UI controls are displayed as long as possible.
    button.pack(side=tkinter.BOTTOM)
    toolbar.pack(side=tkinter.BOTTOM, fill=tkinter.X)
    canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)
    tkinter.Button(neu, text="Ende Neu",command=neu.quit).pack()

# ende function for getting rid of the GUI
def ende():
    main.destroy()

# Get rid of subwindow
def endeneu():
    neu.destroy()

# Verbindung, Cursor
connection = sqlite3.connect("ML_data.db")
cursor = connection.cursor()

# SQL-Abfrage
sql = "SELECT * FROM Data"

# Kontrollausgabe der SQL-Abfrage
# print(sql)

# Absenden der SQL-Abfrage
# Empfang des Ergebnisses
cursor.execute(sql)

# Create Tk object
main = tkinter.Tk()
#Rest of the table
table=[]
for dsatz in cursor:
    table.append(dsatz)



main.wm_title("Accuracy and time dependent on epoch")
e1 = tkinter.Entry(main)
e1.pack()
e2 = tkinter.Entry(main)
e2.pack()
print(e1.get(),e2.get())
tkinter.Button(main, text="Neu", command=fenster).pack()
tkinter.Button(main, text="Ende", command=ende).pack()
main.mainloop()

# Verbindung beenden
connection.close()

