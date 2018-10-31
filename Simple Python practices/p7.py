def is_prime(num):
	if num > 1: 
		if len(a) == 0: 
			print("prime")
		else:
			print("not prime")
	else:
		print("not prime")

again = "yes"
while again == "yes":
	num = int(input("Input a number: "))
	a = [x for x in range(2, num) if num % x == 0]
	is_prime(num)
	again = input("Do you want to go again? write yes if you want to, and no if you don't want to ")