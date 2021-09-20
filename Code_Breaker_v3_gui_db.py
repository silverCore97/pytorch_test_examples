import time, random, glob, sqlite3, \
    tkinter, tkinter.messagebox

# Class for game variables
class Game:
    def __init__(self,name,length=4,rng=4,tries=6):
        self.name=name
        self.length=length
        self.tries=tries
        self.secret=[]
        self.range=rng
        self.it=1
        self.start=0
        self.solved=False
        for i in range(4):
            self.secret.append(random.randint(0,self.range))

    def getName(self):
        return self.name
    def getSecret(self):
        return self.secret
    def getLength(self):
        return self.length

    def getRange(self):
        return self.range

    def getTries(self):
        return self.tries
    def getIt(self):
        return self.it
    def getSolved(self):
        return self.solved


def start():
    name=enname.get()
    # Create game
    game=Game(name)
    game.start=time.time()  #Start time measurement

    # Create game window itself
    window = tkinter.Toplevel(main)

    title = tkinter.Label(window, text="Code breaker: Range from 0 to "+str(game.getRange()))
    title.grid(row=0,column=0,columnspan=2)
    tkinter.Label(window, text="Your name: " +name).grid(row=1,column=0,columnspan=2)
    entries=[]
    for i in range(game.getLength()):
        tkinter.Label(window, text=str(i+1)+". Number:").grid(row=2+i,column=0)
        e=tkinter.Entry(window)
        e.grid(row=2+i,column=1)
        entries.append(e)
    # Buttons of game window
    b1=tkinter.Button(window, text="Check", command=lambda:check(game,entries,window))
    b1.grid(row=3+game.getLength(),column=0)
    b2=tkinter.Button(window, text="Quit", command=window.quit)
    b2.grid(row=3+game.getLength(),column=1)
    #The game progresses by clicking the check button       

    
def check(g,ent,w):
    # guess to list
    glist=[]
    temp=[]
    for entry in ent:
        glist.append(int(entry.get()))
        temp.append("*")
    # Same number and position
    for i in range(0,len(g.getSecret())):
        if glist[i]==g.getSecret()[i]:
            temp[i]="b"

    # If number is another position and not already b
    for i in range(0,len(g.getSecret())):
        if temp[i] !="b":
            for j in range(0,len(g.getSecret())):
                if temp[j] !="b" and glist[i]==g.getSecret()[j]:
                        temp[i]="w"
    print(temp)
    # Check if solution was found
    t=""
    b=""
    for ti in temp:
        t+=ti
        b+="b"
    if t==b:
        g.solved=True

    g.it+=1    #Increase iterator of game

    
    if g.getSolved() or g.it>g.getTries():
        if g.getSolved():
            t2=time.time()  #End time measurement
            diff=t2-g.start
            tkinter.messagebox.showinfo("Game result", "You won!")
            # Highscore-DB nicht vorhanden, erzeugen
            if not glob.glob("highscore_Code_Breaker_gui.db"):
                con = sqlite3.connect("highscore_Code_Breaker_gui.db")
                cursor = con.cursor()
                sql = "CREATE TABLE data(name TEXT, time REAL)"
                cursor.execute(sql)
                con.close()

            # Datensatz in DB schreiben
            con = sqlite3.connect("highscore_Code_Breaker_gui.db")
            cursor = con.cursor()
            sql = "INSERT INTO data VALUES('" \
                + g.name + "'," + str(round(diff,2)) + ")"
            cursor.execute(sql)
            con.commit()
            con.close()
        else:
            tkinter.messagebox.showinfo("Game result", "You lost!")
        w.destroy
    
    tkinter.messagebox.showinfo("Hint", temp)

def highscore():    
    
    
    if not glob.glob("highscore_Code_Breaker_gui.db"):
        tkinter.messagebox.showinfo("Highscore", "No highscore table available yet.")
    else:
        # Highscores sortiert laden
        con = sqlite3.connect("highscore_Code_Breaker_gui.db")
        cursor = con.cursor()
        sql = "SELECT * FROM data ORDER BY time LIMIT 10"
        cursor.execute(sql)

        # Ausgabe Highscore
        result = ""
        i = 1
        for dsatz in cursor:
            result += str(i) + ". " + dsatz[0] + " " \
                + str(round(dsatz[1],2)) + " sec.\n"
            i = i+1
        tkinter.messagebox.showinfo("Highscore", result)
        con.close()
        
def end():
    main.destroy()

def help():
     #Start game description
     description="Welcome to Code Breaker!\n"
     description+="Guess the secret code x. It consists of numbers in a certain range.\n"
     description+="b indicates that the guess for the position is correct.\n"
     description+="w indicates that the number appears somewhere else in the secret code.\n"
     description+="* indicates that the number does not appear in the secret code.\n"
     tkinter.messagebox.showinfo("Help",description)

# Main Program
main = tkinter.Tk()

# Title
lbtitle = tkinter.Label(main, text="Code breaker")
lbtitle.pack()

# Main menu
tkinter.Label(main, text="Your name:").pack()
enname = tkinter.Entry(main)
enname.pack()

tkinter.Button(main, text="Start", command=start).pack(fill="x")

tkinter.Button(main, text="Highscore", command=highscore).pack(fill="x")
tkinter.Button(main, text="Help", command=help).pack(fill="x")
tkinter.Button(main, text="End", command=end).pack(fill="x")







main.mainloop()
