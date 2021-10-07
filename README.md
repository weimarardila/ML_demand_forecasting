# Demand forecasting for inventory management

## Introduction

Inventory is one of the most straightforward ways to add redundancy to protect the supply chain against demand disruptions and unexpected and abrupt changes in demand behavior patterns. The inventory redundancy is a supply chain resilience strategy that allows a system to increase the safety stock levels (usually defined from the variance in the demand forecasts and a security factor related to the normal distribution) to respond in the best way to sudden increases demand. 

In practice, the biggest problem with most supply chain resilience strategies is they suppose a conflict with traditional business goals as efficiency and cost reduction. It is also the case for inventory redundancy. This can be explained since increases in the whole system's inventory levels also imply increases in costs related to inventory management. 

An alternative to deal with this problem is through demand forecasts. These are essential to determining the reorder and safety stock in traditional inventory policies. A better demand forecast, from a statistical learning approach, could generate a better performance in the inventory policy, which would allow the system to react adequately to demand disruptions without unnecessary increases in inventory levels.

## Dataset Description

The initial dataset contains historical product demand for a manufacturing company. The company provides 2160 products distributed in different categories. In addition, there are four central warehouses to ship products to each one of the regions they are responsible for. The dataset is all real-life data and products/warehouse and category information encoded with 1036695 observations collected between 2012 and 2017. 

The database is initially composed of five attributes: Product_Code, Product_Category. Warehouse, Date, Order_Demand.


## Cleaning the Dataset

Before removing outliers from the database, the demand for each product was aggregated per month to better and easier handling of the data and detect patterns in the behavior of the demand. 

To identify and remove atypical data from the database, we calculated the confidence interval for the average weekly demand of each one of the products. Then, we proceeded to eliminate those observations that were outside of that interval.

## Methodolgy

The methodology proposed for this project consisted in the application of three procedures. The first procedure is the model identification, which is focused on the stationarity analysis of the dataset. The second part is the estimation of the parameters, which aims to determine the ARMA or ARIMA coefficients depending on the previous classification. Finally, the prediction and models' performance evaluation make estimations about the future demand and assess the quality of these estimations.

## References

HAli, M., Zied, M., Boylan, J., & Syntetos, A. (2017). Supply chain forecasting when information is not shared. European Journal of Operational Research.

Cheikhrouhou, N., Marmier, F., & Ayadi, O. W. (2011). A collaborative demand forecasting process with event-based fuzzy judgements.

Gilbert, K., & Chatpattananan, V. (2006). An ARIMA suppply chain model with a generalized ordering policy. Journal of modelling in management.

Gunasekaran, A., Papadopoulos, T., Dubey, R., Fosso, S., Childe, S., & Hazen, B. (2017). Big data and predictive analytics for supply chain and organizational perfomance. Journal of Business Research.

Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning Data Mning, Inference, and Prediction. Springer.

Hong, L., & Ping, W. (2007). Bullwhip effect analysis in supply chain for demand forecasting technology. System engineering theory and practice.

Huber, J., Gossmann, A., & Stucknschmidt, H. (2017). Cluster-based hierarchical demand forescasting for perishable goods. Expert Systems with Appplications.

Jaipuria, S., & Mahapatra, S. (2006). An improved demand forecasting method to reduce bullwhip effect in supply chains. Expert Systems with Applications.

Kochak, A., & Sharma, S. (2015). Demand forecasting using neural network for supplu chain management. International Journal of Mechanical Engineering and Robotics Research.

Kourentzes, N., Barrow, D., & Crone, S. (2014). Neural network ensemble operators for time series forecasting. Expert Systems with Applications.

Li, B., Li, J., & Li, W. (2012). Demand forecasting for production planning decision-making based on the new optimized fuzzy short time-series clustering.

Liu, C., Shu, T., Chen, S., Wang, S., Lai, K., & Gan, L. (2016). An improved grey neural network model for predicting transportation disruptions. Expert Systems with Applications.

Murray, P., Agard, B., & Barajas, M. (2015). Forecasting supply chain demand by clustering customers. IFAC, (pp. 1834 - 1839).

Wang, G., Gunasekaran, A., Ngai, E., & Papadopoulos, T. (2016). Big data analytics in logistics and supply chain management. International Journal Production Economics.

Wang, L., Zeng, Y., & Chen, T. (2015). Back propagation neural network with adaptive differential evolution algorithm fot time series forecasting. Expert Systems with Applications.
