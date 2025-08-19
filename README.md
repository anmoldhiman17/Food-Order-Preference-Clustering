# 🍔 Food Order Preference Clustering  

## 📌 Overview  
This project applies **Unsupervised Machine Learning** to cluster food delivery customers based on their **ordering frequency, total spend, and cuisine/restaurant preferences**.  

The goal is to help food delivery platforms (like Zomato/Swiggy) with:  
- 🎯 **Customer Segmentation**  
- 🛒 **Personalized Recommendations**  
- 💸 **Targeted Discounts & Retention Strategies**  

---

## 🚀 Workflow  

1. **Data Preprocessing**  
   - Aggregated data per customer (`Order_Frequency`, `Spend_Amount`, `Cuisine_Type`).  
   - Encoded categorical features using One-Hot Encoding.  
   - Standardized numerical features.  

2. **Dimensionality Reduction (PCA)**  
   - Reduced data into **2 dimensions** for visualization.  

3. **Clustering (K-Means)**  
   - Optimal clusters selected using **Elbow Method**.  
   - Applied **K-Means Clustering**.  
   - Achieved **Silhouette Score = 0.76** → strong, meaningful clusters.  

4. **Visualization & Insights**  
   - Plotted clusters using PCA.  
   - Derived customer insights from cluster summary.  

---

## 📊 Results  

### Identified Customer Segments  

| Cluster | Avg. Orders | Avg. Spend | Segment |
|---------|------------|------------|-----------------------------|
| 0 | ~1.6 | ~₹877 | 🍽 Casual Eaters (low spend, few orders) |
| 1 | ~1.3 | ~₹915 | 🕒 Trial Users / One-time customers |
| 2 | ~14.5 | ~₹10,484 | 👑 VIP Heavy Users (high-value) |
| 3 | ~5.0 | ~₹3,861 | 🔁 Regular Loyal Customers |

---

## 💡 Business Impact  

- 🎯 **Promotions** → Encourage casual users with discounts.  
- 👑 **VIP Loyalty** → Exclusive rewards for heavy spenders.  
- 🔁 **Re-engagement** → Campaigns for one-time users.  
- 🍴 **Personalization** → Cuisine-based recommendations.  

---

## 🛠 Tech Stack  

- **Language:** Python  
- **Libraries:** `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`  
- **Algorithms:** PCA, K-Means Clustering  
- **Environment:** Jupyter Notebook / VS Code  

---

## 📷 Sample Outputs  

- 📈 **Elbow Method Curve** to select optimal clusters  
- 🎨 **PCA Scatter Plot** with 4 distinct clusters  
- 📝 **Cluster Summary Table**  

---
Made with ❤️ by Anmol Dhiman.
