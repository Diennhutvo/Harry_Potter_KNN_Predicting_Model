/*
Project 1: Excelsior Mobile Reportd

Dien Vo

*/

USE [20WQ_BUAN4210_Lloyd_ExcelsiorMobile];

--Report Question with Visualization 

--A

SELECT CONCAT( FirstName, ' ', LastName) AS 'Full Name', Minutes, DataInMB, Texts, Total 
FROM Subscriber, Bill, LastMonthUsage
WHERE Subscriber.MIN = Bill.MIN 
AND Subscriber.MIN = LastMonthUsage.MIN
ORDER BY 'Full Name';

/* first and last names of our customers along with their
 minute usage, data usage, text usage and total bill. Order them by their full name.*/

--B

SELECT City, AVG(Minutes) AS ' AVG Mins', AVG (DataInMB) AS 'AVG Data', AVG( Texts) AS 'AVG Texts' ,AVG (Total) AS ' AVG Total'
FROM Subscriber, Bill, LastMonthUsage
WHERE Subscriber.MIN = BILL.MIN 
AND Subscriber.MIN = LastMonthUsage.MIN
GROUP BY City
ORDER BY City;

-- the average of the minutes, data, texts and total bills by city.

--C

SELECT City, SUM(Minutes) AS ' SUM Mins', SUM (DataInMB) AS 'SUM Data', SUM( Texts) AS 'SUM Texts' , SUM(Total) AS ' SUM Total' 
FROM Subscriber, Bill, LastMonthUsage
WHERE Subscriber.MIN = BILL.MIN 
AND Subscriber.MIN = LastMonthUsage.MIN
GROUP BY City
ORDER BY City;

--the sum of the minutes, data, texts and total bills by city.

--D

SELECT PlanName, AVG(Minutes) AS ' AVG Mins', AVG (DataInMB) AS 'AVG Data', AVG( Texts) AS 'AVG Texts' , AVG(Total) AS 'AVG Total'
FROM Subscriber, Bill, LastMonthUsage
WHERE Subscriber.MIN = BILL.MIN 
AND Subscriber.MIN = LastMonthUsage.MIN
GROUP BY PlanName
ORDER BY PlanName;

--the average of the minutes, data, texts and total bills by mobile plan.

--E

SELECT PlanName, SUM(Minutes) AS ' SUM Mins', SUM (DataInMB) AS 'SUM Data', SUM( Texts) AS 'SUM Texts' , SUM(Total) AS 'SUM Total'
FROM Subscriber, Bill, LastMonthUsage
WHERE Subscriber.MIN = BILL.MIN 
AND Subscriber.MIN = LastMonthUsage.MIN
GROUP BY PlanName
ORDER BY PlanName;

--the sum of the minutes, data, texts and total bills by mobile plan.

-- REPORT QUESTION WITHOUT VISUALIZATION 

--1 

--A 

SELECT TOP 2 City, COUNT(City) AS 'Number of customers' 
FROM Subscriber
GROUP BY City
ORDER BY COUNT(City) DESC;

--which two cities we have the most customers in.

--B 

SELECT City, COUNT(City) AS 'Number of customers' 
FROM Subscriber
GROUP BY City
HAVING COUNT(City) <4
ORDER BY COUNT(City) ASC;

--which cities we should increase our marketing in.

--C

SELECT  PlanName, COUNT(PlanName) AS 'Number of customers using Plan' 
FROM Subscriber
GROUP BY PlanName
ORDER BY COUNT(City) DESC;

--which plans we should market the most based on the number of people who have them 

--2 

--A

SELECT Type, COUNT(Type) AS ' Number of customers use' 
FROM Device
GROUP BY Type 
ORDER BY COUNT (Type) DESC;

/*the count of cell phone types among our customers.
 What type do most of our customers use?*/

--B

SELECT  Type,CONCAT( FirstName, ' ', LastName) AS 'Full Name'
FROM Subscriber,Device , DirNums
WHERE Subscriber.MDN = DirNums.MDN
AND Device.IMEI = DirNums.IMEI
GROUP BY Type, FirstName, LastName
HAVING Type = 'Apple';

--C 

SELECT CONCAT( FirstName, ' ', LastName) AS 'Full Name', YearReleased
FROM Subscriber, DirNums, Device
WHERE Subscriber.MDN = DirNums.MDN 
AND DirNums.IMEI = Device.IMEI
GROUP BY  FirstName, LastName, YearReleased 
HAVING YearReleased < 2018 
ORDER BY YearReleased;

--our customers and the year of their phones who have phones released before the last two years?

--3


SELECT  TOP 3 DataInMB,MDN, City, Data
FROM Subscriber, MPlan, LastMonthUsage
WHERE Subscriber.PlanName = MPlan.PlanName
AND Subscriber.MIN = LastMonthUsage.MIN 
GROUP BY  DataInMB,MDN,City, Data
HAVING Data NOT LIKE '%Unlimited'
ORDER BY DataInMB DESC; 

--if there is a city that uses a lot of data (within the top 3 data using cities) but none of our customers in that city are using the Unlimited Plans.

--4

--A

SELECT TOP 1 Total,CONCAT( FirstName, ' ', LastName) 'Full Name'
FROM Subscriber, Bill
WHERE Subscriber.MIN = BILL.MIN
GROUP BY FirstName, LastName,Total
ORDER BY Total DESC;

--first and last name of the customer who has the most expensive bill every month.

--B

SELECT PlanName, SUM(Total) AS 'Revenue'
FROM Subscriber, Bill
WHERE Subscriber.MIN = BILL.MIN
GROUP BY PlanName
ORDER BY SUM(Total) DESC;

--which mobile plan brings us in the most revenue each month

--5

--A

SELECT   TOP 1 Minutes, ZipCode 
FROM Subscriber, LastMonthUsage
WHERE Subscriber.MIN = LastMonthUsage.MIN
ORDER BY Minutes DESC; 

--which area code (only the area code) uses the most minutes.

--B

SELECT City,Minutes 
FROM Subscriber, LastMonthUsage 
WHERE Subscriber.MIN = LastMonthUsage.MIN 
GROUP BY City, Minutes
HAVING Minutes < 200 
OR Minutes > 700;

--Which cities are in the most variance in terms of minutes usage?