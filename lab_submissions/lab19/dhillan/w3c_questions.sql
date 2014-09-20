
-- What customers are from the UK?
SELECT *
FROM Customers
WHERE Country = 'UK'

-- Find 2 customers from the UK.
SELECT *
FROM Customers
WHERE COUNTRY = 'UK'
LIMIT 2

-- What are the ids and names of the customers with the 10 smallest CustomerIDs?
SELECT CustomerID, CustomerName
FROM Customers
ORDER BY CustomerID
LIMIT 10

-- What are the ids and names of the customers with the 10 highest CustomerIDs?
SELECT CustomerID, ContactName
FROM Customers
ORDER BY CustomerID DESC
LIMIT 10

-- How many customers are from each country?
SELECT COUNT(CustomerID), Country
FROM Customers
GROUP BY Country

-- What country provides the most customers?
SELECT COUNT(CustomerID), Country
FROM Customers
GROUP BY Country
ORDER BY COUNT(CustomerID) DESC
LIMIT 1

-- What is the id of the customer who has the most orders? 
SELECT CustomerID
FROM [Orders]
GROUP BY CustomerID
ORDER BY COUNT(OrderID) DESC
LIMIT 1

-- What is the name of the customer who has the most orders? 
SELECT Customers.CustomerName
FROM Orders JOIN Customers ON Orders.CustomerID = Customers.CustomerID
GROUP BY Orders.CustomerID
ORDER BY COUNT(OrderID) DESC
LIMIT 1

-- What are the names of the customers who only have one order? 
SELECT Customers.CustomerName, COUNT(Orders.OrderID)
FROM Orders JOIN Customers ON Orders.CustomerID = Customers.CustomerID
GROUP BY Orders.CustomerID
HAVING COUNT(Orders.OrderID) = 1;

-- What was the total price of OrderID = 10253? 
SELECT SUM(OrderDetails.Quantity *Products.Price)
FROM Orders
INNER JOIN OrderDetails, Products
ON Orders.OrderID=OrderDetails.OrderID AND OrderDetails.ProductID = Products.ProductID
WHERE Orders.OrderID = 10253;

-- What supplier has the highest average product price?
SELECT Suppliers.SupplierName
FROM Products
INNER JOIN Suppliers
ON Products.SupplierID = Suppliers.SupplierID
GROUP BY Products.SupplierID
ORDER BY AVG(Products.Price) DESC
LIMIT 1
-- What supplier has the highest average product price and more than 20 products?
-- Couldn't get this one. Do any suppliers have more than 20 products?
SELECT *, AVG(Products.Price) as AvgPrice
FROM Products
INNER JOIN Suppliers
ON Products.SupplierID = Suppliers.SupplierID
GROUP BY Products.SupplierID
HAVING Count(Products.ProductID) >= 2
ORDER BY AVG(Products.Price) DESC
LIMIT 1
-- What employees have BS degrees? 
SELECT * FROM [Employees]
WHERE Notes LIKE '%BS%'