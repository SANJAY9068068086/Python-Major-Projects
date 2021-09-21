#!/usr/bin/env python
# coding: utf-8

# # Simple Shopping Python Program for Beginners

# In[ ]:


stock = {'Apple' : 100, 'Banana' : 150, 'Orange' : 80, 'Pear' : 200, 'Guava' : 30, 'Mango' : 60}
prices = {'Apple' : 120, 'Banana' : 55, 'Orange' : 40, 'Pear' : 45, 'Guava' : 25, 'Mango' : 150}

def uppercase(x):
    return x[0].upper()+x[1:]

name = input("What is your Name : ")
print("Hi %s, Welcome to Fruit Store. Here is the menu : "%(name))
print()

def menu():
    for fruit in prices:
        print(uppercase(fruit),"\n--------")
        print("Stock  : %s"%stock[fruit])
        print("Prices : Rs.%s"%prices[fruit])
        print()
    print("You have: Rs. %s"%(money))
    print()
    
def ask_fruit(money):
    fruit = input("What fruit do you want? : ")
    print()
    if fruit in stock:
        if stock[fruit]>0:
            ask_amount(fruit,money)
        else:
            print("Sorry, %s's are out of stock"%(fruit))
            ask_fruit(money)
    else:
        print("Sorry, We don't have that, look at the menu.")
        ask_fruit(money)
        
def ask_amount(fruit,money):
    amount = int(input("How many %s's do you want? : "%(fruit)))
    print()
    if amount<=0:
        print("At least buy one.")
        ask_amount(fruit,money)
    elif stock[fruit]>=amount:
        sell(fruit,amount,money)
    else:
        print("Sorry We don't have that many %s's"%(fruit))
        ask_amount(fruit,money)

def sell(fruit,amount,money):
    cost = prices[fruit]*amount
    confirmation = input('''Are you sure? That will be Rs.%s
    -YES
    -NO\n'''
    %(cost)).lower()
    print()
    if confirmation == 'yes':
        money-=cost
        print("Thank You for the business!")
        stock[fruit]=stock[fruit]-amount
        ask_again()
    elif confirmation == 'no':
        ask_fruit(money)
    else:
        print("Answer Me.")
        sell(fruit,amount,money)
    
def ask_again():
    answer = input('''Do you want anything else?
    -YES
    -NO\n''').lower()
    print()
    if answer ==  'yes':
        menu()
        ask_fruit(money)
    elif answer == 'no':
        print("Okey, Bye.")
    else:
        print("Answer Me.")
        ask_again()
money = 117
menu()
ask_fruit(money)


# In[ ]:




