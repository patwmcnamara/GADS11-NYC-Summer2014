
-- What customers are from the UK?
SELECT *
FROM Customers
WHERE Country = 'UK'

-- What is the name of the customer who has the most orders?
SELECT *
FROM Customers
WHERE Country = 'UK'
LIMIT 2

-- What are the ids and names of the customers with the 10 smallest CustomerIDs?
SELECT CustomerID, CustomerName
FROM Customers
ORDER BY CustomerID
LIMIT 10

-- What are the ids and names of the customers with the 10 highest CustomerIDs?
SELECT CustomerID, CustomerName
FROM Customers
ORDER BY CustomerID DESC
LIMIT 10

-- How many customers are from each country?
SELECT Country, Count(*)
FROM Customers
GROUP BY Country

-- What country provides the most customers?
SELECT Country
FROM Customers
GROUP BY Country
ORDER BY Count(*) DESC
LIMIT 1

-- What is the id of the customer who has the most orders?
SELECT CustomerID
FROM Orders
GROUP BY CustomerID
ORDER BY Count(*) DESC
LIMIT 1

-- What is the name of the customer who has the most orders?
SELECT CustomerName
FROM Orders o
JOIN Customers c ON c.CustomerID = o.CustomerID
GROUP BY c.CustomerID
ORDER BY Count(*) DESC
LIMIT 1

-- What are the names of the customers who only have one order?
SELECT CustomerName
FROM Orders o
JOIN Customers c ON c.CustomerID = o.CustomerID
GROUP BY c.CustomerID
HAVING Count(*) = 1

-- What was the total price of OrderID = 10253?
SELECT Sum(o.Quantity * p.Price)
FROM OrderDetails o
JOIN Products p ON p.ProductID = o.ProductID
WHERE OrderID = 10253

-- What supplier has the highest average product price?
SELECT s.SupplierName
FROM Suppliers s
JOIN Products p ON p.SupplierID = s.SupplierID
GROUP BY s.SupplierID
ORDER BY Avg(p.Price) DESC
LIMIT 1

-- What supplier has the highest average product price and more than 2 products?
SELECT s.SupplierName
FROM Suppliers s
JOIN Products p ON p.SupplierID = s.SupplierID
GROUP BY s.SupplierID
HAVING Count(p.ProductID) > 2
ORDER BY Avg(p.Price) DESC
LIMIT 1

-- What employees have BS degrees?
SELECT * FROM Employees WHERE Notes LIKE '%BS%'
