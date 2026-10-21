\# Date Arithmetic in Prolog (Variant 1)



This project implements a Prolog program to perform simple addition and subtraction of dates by an integer number of days ($N$) for the leap year 2024.



\## Prerequisites

To run this code, you need to have \*\*SWI-Prolog\*\* installed on your system, or you can use the web-based version (https://swish.swi-prolog.org/)



\## Queries 



1. Date Addition (add\_date)Adds $N$ days to a given date string in "DDMM" add\_date(+DateStr, +N).



Example:

add\_date("2205", 14).

"0506"

true.



2\. Date Subtraction (sub\_date)Subtracts $N$ days from a given date string in "DDMM" 

&#x20;  sub\_date(+DateStr, +N)



Example:

sub\_date("2205", 10).

"1205"

true.

