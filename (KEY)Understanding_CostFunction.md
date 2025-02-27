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
- $$\( \hat{y}_{ij} \)$$ = predicted probability for class \( j \)  

### **Why Use Cross-Entropy?**  
- Works well with **softmax** for multi-class classification.  
- Punishes incorrect confident predictions more.  
- Ensures gradients are well-behaved for **optimization**.  

Thus, **Cross-Entropy Loss is a cost function** used for training classification models.
- If Test and Train Model aint close to each other, means Train Model is overfitt.
