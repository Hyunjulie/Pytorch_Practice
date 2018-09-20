#Get a word from user and evaluate if the word is palindrome (대칭)

import string

usr_inp = str(input("Type in any word: "))
rev = usr_inp[::-1]
print(rev)

if usr_inp == rev:
	print("This word is palindrome")
else: 
	print("This word is not palindrome")


# One line of code to print out only even elements in a list
# a = [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]
# print([i for i in a if i % 2 == 0])
