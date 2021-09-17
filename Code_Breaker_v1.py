# Single round codebreaker
import random

# check takes the secret in list form and the guess as Integer
def check(secret,guess):
    # guess to list
    glist=[]
    temp=[]
    for i in range(0,len(secret)):
        glist.append(int((guess/(pow(10,3-i)))%10))
        temp.append("*")
    print(glist)
    # Same number and position
    for i in range(0,len(secret)):
        if glist[i]==secret[i]:
            temp[i]="b"

    # If number is another position and not already b
    for i in range(0,len(secret)):
        if temp[i] !="b":
            for j in range(0,len(secret)):
                if temp[j] !="b" and glist[i]==secret[j]:
                        temp[i]="w"

    return temp
#Start game
print("Welcome to Code Breaker!")
print("Guess the secret code x. It consists of 4 numbers between 1 and 4.\n")
print("b indicates that the guess for the position is correct.\n"
      "w indicates that the number appears somewhere else in the secret code.\n"
      "* indicates that the number does not appear in the secret code.\n\n")

#Initialize secret code
random.seed()
secret=[]
for i in range(0,4):
    secret.append(random.randint(1,4))

# 10 Tries
j=0
temp=""
while j<9:
    print("Your "+str(j+1)+". guess: ")
    try:

        guess=int(input())

    except:
        print("Incorrect answer type.")
        continue
    if len(secret) == len(str(guess)):
        temp = check(secret, guess)
        x=""
        for i in range(0, len(secret)):
            x=x+temp[i]
        print(x+"\n")
        if "*" not in temp and "w" not in temp:
            print("\nYou found the Code!")
            break;
    else:
        print("Incorrect answer length")
    #Increase Iterator
    j+=1

#If unsuccessgul
if "*" in temp or "w" in temp:
    print("\nYou were unsuccessful in finding the code:" +secret)
