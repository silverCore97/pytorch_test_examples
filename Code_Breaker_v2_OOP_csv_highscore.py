import random, time, glob

# Klasse "Spiel"
class Spiel:
    def __init__(self):
        # Start game
        random.seed()
        self.secret=[]
        for i in range(0,4):
            self.secret.append(random.randint(1,4))
        self.max_tries=10
        self.curr=1
        s = input("Please enter your name (Max 10 chars)")
        self.spieler = s[0:10]
        self.solved=False

    def spielen(self):
        #Start game description
        print("Welcome to Code Breaker!")
        print("Guess the secret code x. It consists of 4 numbers between 1 and 4.\n")
        print("b indicates that the guess for the position is correct.\n"
              "w indicates that the number appears somewhere else in the secret code.\n"
              "* indicates that the number does not appear in the secret code.\n")

        #Actual game
        for self.curr in range(1,self.max_tries+1):
            a = Aufgabe(self.secret,self.curr, self.max_tries)
            result=str(a)
            print(result +"\n")
            
            if result == "bbbb":
                self.solved=True
                break

    def messen(self, start):
        # Time measurement
        if start:
            self.startzeit = time.time()
        else:
            endzeit = time.time()
            self.zeit = endzeit - self.startzeit

    def __str__(self):
        # Ergebnis
        ausgabe =  f"{self.zeit:.2f} Sekunden"
        if self.solved:
            ausgabe += ", Highscore"
            hs = Highscore()
            hs.speichern(self.spieler, self.zeit)
            print(hs)
        else:
            ausgabe += ", leider kein Highscore"
        return ausgabe

# Class "Aufgabe"
class Aufgabe:
    # Aufgabe initialisieren
    def __init__(self, secret,i, max):

        self.nr = i
        self.max=max
        self.secret=secret

    # Aufgabe stellen
    def __str__(self):
        temp=""
        if self.nr==self.max:
            temp="Last "
        else:
            temp=str(self.nr)+". "
        temp+="try:"
        print( temp)
        res= self.beantworten()
        return res
        
    # Aufgabe beantworten
    def beantworten(self):
        try:
            guess=int(input())
        except:
            return "incorrect answer type"
            
        if len(self.secret) == len(str(guess)):
            temp = self.check(guess)
            string =""
            for i in range(len(temp)):
                string+=str(temp[i])
            return string
        else:
            return "Incorrect answer length"
            

    #Check method to determine correctness of check()
    def check(self,guess):
        # guess to list
        glist=[]
        temp=[]
        for i in range(0,len(self.secret)):
            glist.append(int((guess/(pow(10,3-i)))%10))
            temp.append("*")
        # Same number and position
        for i in range(0,len(self.secret)):
            if glist[i]==self.secret[i]:
                temp[i]="b"

        # If number is another position and not already b
        for i in range(0,len(self.secret)):
            if temp[i] !="b":
                for j in range(0,len(self.secret)):
                    if temp[j] !="b" and glist[i]==self.secret[j]:
                            temp[i]="w"
        return temp

# Klasse "Highscore"
class Highscore:
    # Liste aus Datei lesen
    def __init__(self):
        self.liste = []
        if not glob.glob("highscore_code_breaker.csv"):
            return
        d = open("highscore_code_breaker.csv")
        zeile = d.readline()
        while(zeile):
            teil = zeile.split(";")
            name = teil[0]
            zeit = teil[1][0:len(teil[1])-1]
            zeit = zeit.replace(",", ".")
            self.liste.append([name, float(zeit)])
            zeile = d.readline()
        d.close()

    # Liste ändern
    def aendern(self, name, zeit):
        # Mitten in Liste schreiben
        gefunden = False
        for i in range(len(self.liste)):
            # Einsetzen in Liste
            if zeit < self.liste[i][1]:
                self.liste.insert(i, [name, zeit])
                gefunden = True
                break

        # Ans Ende der Liste schreiben
        if not gefunden:
            self.liste.append([name, zeit])

    # Liste ändern, in Datei speichern
    def speichern(self, name, zeit):
        self.aendern(name, zeit)
        d = open("highscore_code_breaker.csv", "w")
        for element in self.liste:
            name = element[0]
            zeit = str(element[1]).replace(".", ",")
            d.write(name + ";" + zeit + "\n")
        d.close()

    # Liste anzeigen
    def __str__(self):
        # Highscore nicht vorhanden
        if not self.liste:
            return "Keine Highscores vorhanden"

        # Ausgabe Highscore
        ausgabe = " P. Name            Zeit\n"
        for i in range(len(self.liste)):
            ausgabe += f"{i+1:2d}. {self.liste[i][0]:10}" \
                       f"{self.liste[i][1]:5.2f} sec\n"
            if i >= 9:
                break
        return ausgabe

# Main menu
while True:
    # Choose option
    try:
        menu = int(input("Please enter "
            "(0: End, 1: Highscores, 2: Play): "))
    except:
        print("Wrong entry")
        continue

    # Object creation or end
    if menu == 0:
        break
    elif menu == 1:
        hs = Highscore()
        print(hs)
    elif menu == 2:
        s = Spiel()
        s.messen(True)
        s.spielen()
        s.messen(False)
        print(s)
    else:
        print("Falsche Eingabe")
