#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Simple input Dictionary Program for Beginners

keys = []
values = []

dict_range = int(input("Enter the range of your dictionary : "))
for i in range(dict_range):
    x = input("Enter Key : ")
    keys.append(x)
    y = input("Enter Value : ")
    values.append(y)
    
my_dict = dict(zip(keys,values))
print()    
find = input("To find the Value, Please Enter Key : ")
print("\n","-"*50,"\nValue of Entered Key ",[find]," is : ",my_dict[find],sep="")


# In[ ]:




