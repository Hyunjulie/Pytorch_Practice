import sys 

usr1 = input("What's your name? ")
usr2 = input("And your name is?")

usr1_answer = input("%s, what is your choice? Rock, Paper Scissors: ")
usr2_answer = input("%s, and your choice is? ")

def compare(u1, u2):
	if u1 == u2: 
		return("It's a tie!")
	elif 