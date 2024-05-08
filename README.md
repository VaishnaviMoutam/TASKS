## PRINTING OF FIRST 10 NUMBERS

## This is the code explanation about printing of first 10 numbers and it performs the following tasks:

num = list(range(10))

It creates a list named 'num' containing numbers from 0 to 9 using the 'range()' function.

previousNum = 0

It initializes a variable 'previousNum' with a value of 0.

for i in num:

It iterates through each element 'i' in the 'num' list.

    sum = previousNum + i
    Inside the loop, it calculates the sum of the current number 'i' and the previous number 'previousNum'.
    
    print('Current Number '+ str(i) + 'Previous Number ' + str(previousNum) + 'is ' + str(sum))
    It prints the current number 'i', the previous number 'previousNum', and their sum 'sum' as a formatted string.
    
    previousNum=i   
    It updates the value of 'previousNum' to the current number 'i' for the next iteration.

## After all this we get output:

Current Number 0Previous Number 0is 0
Current Number 1Previous Number 0is 1
Current Number 2Previous Number 1is 3
Current Number 3Previous Number 2is 5
Current Number 4Previous Number 3is 7
Current Number 5Previous Number 4is 9
Current Number 6Previous Number 5is 11
Current Number 7Previous Number 6is 13
Current Number 8Previous Number 7is 15
Current Number 9Previous Number 8is 17

##  HISTOGRAM OF AN IMAGE

## What is a Histogram?

A histogram is a graphical representation of the distribution of numerical data. It consists of a series of adjacent rectangles, or bins, each representing a range of data values, and the height of each rectangle corresponds to the frequency of data points falling within that range.






    
