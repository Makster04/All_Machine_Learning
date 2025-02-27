Yes, **Mean Squared Error (MSE)** is a type of **cost function** used primarily in **regression models**.  

### **Why is MSE a Cost Function?**  
A **cost function** measures how far a model's predictions are from the actual values. MSE calculates this by:  
1. Squaring the difference between predicted and actual values.  
2. Averaging these squared differences over all data points.  

### **MSE Formula:**  
$$\[
MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
\]$$
where:  
- $$\( y_i \)$$ = actual value  
- $$\( \hat{y}_i \)$$ = predicted value  
- $$\( n \)$$ = total number of observations  

### **Why Squaring?**  
- Punishes **larger errors** more than smaller ones.  
- Ensures the function is differentiable for **gradient descent** optimization.  

Thus, MSE is commonly used as a **cost function** in regression tasks.

Yes, **Cross-Entropy Loss** is a type of **cost function** used primarily in **classification problems**.  

### **Why is Cross-Entropy a Cost Function?**  
A **cost function** measures how well a model's predicted probabilities match the actual labels. **Cross-Entropy Loss** does this by:  
1. Comparing the predicted probability distribution with the actual class labels.  
2. Penalizing incorrect predictions more heavily when confidence is high.  

### **Binary Cross-Entropy (for two classes):**  
$$\[
\text{Loss} = - \frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]
\]$$
where:  
- $$\( y_i \)$$ = actual label (0 or 1)  
- $$\( \hat{y}_i \)$$= predicted probability  
- $$\( n \)$$ = number of samples  

### **Categorical Cross-Entropy (for multiple classes):**  
$$\[
\text{Loss} = - \sum_{i=1}^{n} \sum_{j=1}^{k} y_{ij} \log(\hat{y}_{ij})
\]$$
where:  
- $$\( k \)$$ = number of classes  
- $$\( y_{ij} \)$$ = actual class (one-hot encoded)  
- $$\( \hat{y}_{ij} \)$$ = predicted probability for class $$\( j \)$$

### **Why Use Cross-Entropy?**  
- Works well with **softmax** for multi-class classification.  
- Punishes incorrect confident predictions more.  
- Ensures gradients are well-behaved for **optimization**.  

Thus, **Cross-Entropy Loss is a cost function** used for training classification models.
- If Test and Train Model aint close to each other, means Train Model is overfitt.
