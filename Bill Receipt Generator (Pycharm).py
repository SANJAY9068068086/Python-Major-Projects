# Bill Receipt Generator Program

class receipt_generator():
    def mainmenu():
        sum = 0
        sum_base = 0
        sum_gst = 0
        Iteams = {}
        print("\nSelect Type of Customer :- \n\nPress 1 for Individual Customer\nPress 2 for GST Customer",sep="")
        type = int(input("Please Enter : "))
        if type == 1:
            bill_no = int(input("Enter Bill Number  : "))
            recipient = input("Enter the  Customer Name : ")
            address = input("Enter the address : ")
            contact = int(input("Enter the contact number : "))
            while True:
                UserChoice = input("Press Y for continue and Press any key to stop : ")
                if UserChoice == 'Y':
                    UserItemName = input("Enter the item name : ")
                    UserQty = float(input("Enter the Quantity : "))
                    UserInput = float(input("Enter the item price : "))
                    UserTotal = UserQty * UserInput
                    UserTotal_gst = (UserQty * ((UserInput * 18) / 118))
                    UserTotal_base = (UserQty * ((UserInput) - ((UserInput * 18) / 118)))
                    sum = sum + UserTotal
                    sum_base = sum_base + UserTotal_base
                    sum_gst = sum_gst + UserTotal_gst
                    print(f"Order total so far : {sum}", sep="")
                    Iteams[UserItemName] = UserTotal_base
                else:
                    print("=" * 50, '\n                G.G Traders Pvt. Ltd.\n', "=" * 50, sep="")
                    print("GST No. 06BUQPS2745R1ZK              Bill No.", bill_no)
                    print("\n                --Your Billing Details--\n")
                    print("Customer Name : ", recipient)
                    print("Contact Number : ", contact)
                    print("Address : ", address, "\n\n")
                    for keys, values in Iteams.items():
                        print(keys, " : ", values, sep="")
                    print(f"\nYour Total bill : {sum_base}"
                          f"\nTotal GST 18% : {sum_gst}"
                          f"\nGrand Total Bill : {sum}"
                          f"\n\nThanks for shopping with us.\n", 50 * "-", sep="")
                    break
        if type == 2:
            bill_no = int(input("Enter Bill Number  : "))
            recipient = input("Enter the  Firm Name : ")
            address = input("Enter the address : ")
            contact = int(input("Enter the contact number : "))
            while True:
                UserChoice = input("Press Y for continue and Press any key to stop : ")
                if UserChoice == 'Y':
                    UserItemName = input("Enter the item name : ")
                    UserQty = float(input("Enter the Quantity : "))
                    UserInput = float(input("Enter the item price : "))
                    UserTotal = UserQty * UserInput
                    UserTotal_gst = (UserQty * ((UserInput*18)/118))
                    UserTotal_base = (UserQty * ((UserInput)-((UserInput*18)/118)))
                    sum = sum + UserTotal
                    sum_base = sum_base + UserTotal_base
                    sum_gst = sum_gst + UserTotal_gst
                    print(f"Order total so far : {sum}",sep="")
                    Iteams[UserItemName]=UserTotal_base
                else:
                    print("="*50,'\n                G.G Traders Pvt. Ltd.\n',"="*50,sep="")
                    print("GST No. 06BUQPS2745R1ZK              Bill No.",bill_no)
                    print("\n                --Your Billing Details--\n")
                    print("Firm Name : ",recipient)
                    print("Contact Number : ",contact)
                    print("Address : ",address,"\n\n")
                    for keys, values in Iteams.items():
                        print(keys, " : ", values,sep="")
                    print(f"\nYour Total bill : {sum_base}"
                          f"\nTotal GST 18% : {sum_gst}"
                          f"\nGrand Total Bill : {sum}"
                          f"\n\nThanks for shopping with us.\n",50*"-",sep="")
                    break
        else:
            print("You Entered wrong option. Please try again")
ob=receipt_generator
ob.mainmenu()