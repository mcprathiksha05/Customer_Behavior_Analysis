# Customer Behavior Analysis Using Python

## 1. Abstract
This project demonstrates how data science can be used to understand customer purchasing behavior. It cleans transaction records, calculates sales metrics, visualizes trends, groups customers by spending, and produces actionable business insights.

## 2. Objectives
- Clean and validate customer order data.
- Measure sales, order volume, customer count, units sold, and average order value.
- Analyze sales by month, category, region, and purchase channel.
- Segment customers by total spending.
- Generate recommendations based on observed patterns.

## 3. Tools and Technologies
Python, Pandas, NumPy, Matplotlib, Seaborn, and Streamlit.

## 4. Methodology
1. Load a CSV dataset.
2. Remove exact duplicate records.
3. Parse order dates and convert quantity and unit price to numeric values.
4. Remove records with missing essential identifiers, dates, quantities, or prices.
5. Remove invalid quantities and negative prices.
6. Fill missing descriptive fields with `Unknown`; impute missing satisfaction scores with the median.
7. Calculate sales using `Sales = Quantity × Unit_Price`.
8. Group and visualize sales by month, category, region, channel, and customer.
9. Segment customers into low-, medium-, and high-spending groups using spending terciles.
10. Export cleaned data and a summary of insights.

## 5. Dashboard Features
- Interactive filters for date, region, category, and purchase channel.
- KPI cards for total sales, unique orders, unique customers, units sold, and average order value.
- Monthly sales line chart.
- Category sales bar chart.
- Purchase-channel sales chart.
- Regional order chart.
- Customer spending and frequency analysis.
- Downloads for cleaned records, customer summaries, and insight reports.

## 6. Business Recommendations
- Reward high-value customers with loyalty benefits.
- Encourage medium-spend customers to make repeat purchases through relevant offers.
- Review low-performing product categories and channels before reallocating marketing spend.
- Track satisfaction alongside repeat purchase behavior.
- Refresh analysis regularly to identify changes in customer behavior.

## 7. Testing
Use the checklist in `README.md`. Validate the dashboard with the supplied synthetic dataset, then test with an authorized CSV containing the required columns.

## 8. Limitations
The included data is synthetic and intended only for demonstrating the workflow. Any findings from it are examples, not claims about a real business. Real business recommendations should be based on authorized, representative data.

## 9. Conclusion
The project provides an end-to-end customer analytics workflow from data cleaning to interactive visualization and report generation. It demonstrates practical use of Python data-science libraries and creates an extensible foundation for further work such as RFM analysis, churn prediction, and cohort analysis.
