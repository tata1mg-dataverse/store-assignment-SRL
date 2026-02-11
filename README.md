# 📦 Delivery Data – README

## 📖 Overview

This dataset contains order-level delivery information used for Store Allocation Project.

---

## 📊 Column Descriptions

### 1. GROUP_ID  
Main order ID.  
Represents the parent order under which multiple individual orders may exist.

### 2. Order_id  
Unique identifier for each individual order.

### 3. TAT (Turn Around Time)  
Actual time taken to deliver the product to the customer.

### 4. ETA (Estimated Time of Arrival)  
Estimated delivery time provided at the time of order placement.

### 5. Central_Pincode  
Pincode of the customer’s delivery location.

---

## 🚚 Delivery Classification

Delivery speed is determined based on **TAT**:

- **Fast Delivery**: Delivered within minutes to hours  
- **Slow Delivery**: Took 1 or more days to deliver  


---
