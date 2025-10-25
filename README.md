# Project Title: Citi Bike Trip Histories 

# Project Description

This project explores the Citi Bike system data from New York City.
Citi Bike is a public bicycle sharing program where people rent bikes for short rides.
The goal of this project is to understand how people use these bikes, such as when they ride, which stations they start from, and how long their trips last.

The data is freely available from the official website:
👉 https://citibikenyc.com/system-data

Each file contains details about millions of trips taken by people in New York City.
Because the raw data files are very large, they were cleaned and saved in a special format called **Parquet** which is faster and smaller in size compared to normal CSV files.

This project teaches how to handle large real-world datasets and draw useful conclusions from them using simple Python tools.


## Objectives
1. Learn how to work with **big datasets** that contain millions of rows.  
2. Study **riding behavior** — when people ride, where they ride, and how long.  
3. Clean and organize raw data so it is easier to study.  
4. Convert repeated text values into efficient data types to save memory.  
5. Generate basic statistics and meaningful insights from the data.  
6. Build a foundation for future visualization or machine learning projects.

Each of these objectives helps improve data handling, thinking, and analytical skills.

## Dataset Description
The Citi Bike dataset records every single bike trip.  
Each row represents **one ride**.  
Below are the important columns and what they mean:

| Column Name | Description |
|--------------|-------------|
| `ride_id` | A unique number for each trip. It is only for identification. |
| `rideable_type` | The kind of bike used — classic, electric, or docked. |
| `started_at` / `ended_at` | Date and time when the ride began and ended. |
| `start_station_name` / `end_station_name` | Names of the stations where the trip started and ended. |
| `start_lat` / `start_lng` | Geographic coordinates of the start station. |
| `end_lat` / `end_lng` | Geographic coordinates of the end station. |
| `member_casual` | The type of user — “member” means a subscribed user, “casual” means a one-time user. |

Sometimes older data files contain extra information such as rider gender or birth year.  
Those columns may be missing in newer files because the data structure changes over time.

In this project, columns like `ride_id`, `start_station_id`, and `end_station_id` were **removed** because they do not help in understanding user behavior.  
They are just technical identifiers and make the file heavier.

## Key Ideas and Insights
After cleaning and organizing the data, we can find many interesting facts, such as:

1. **Station Usage:**  
   Find which start and end stations have the most rides.  
   This helps know where bikes are most needed.

2. **Time Patterns:**  
   Analyze which hours and days of the week have the highest demand.  
   For example, morning and evening peaks may show work commute times.

3. **Member vs Casual Behavior:**  
   Compare how subscribed members and casual users ride differently.  
   Members might take shorter daily trips while casual riders may explore on weekends.

4. **Trip Duration Trends:**  
   Calculate the average ride time and see how it changes by day or season.

5. **Geographic Movement:**  
   Use the station locations (latitude and longitude) to map travel patterns around New York City.

6. **Performance Optimization:**  
   Use smaller and smarter data types so the analysis runs faster and uses less computer memory.

These insights can later be used by city planners or the company to improve the bike system.


##  Tools and Technologies

| Tool | What It Is Used For |
|------|----------------------|
| **Python** | Main language for analysis. |
| **Pandas** | Helps read, clean, and explore data easily. |
| **Google Colab** | Free online tool to write and run Python code in the browser. |
| **Parquet File Format** | Used to store cleaned data quickly and save space. |
| **Matplotlib / Seaborn** | Used to create charts and graphs for visualization. |
| **Citi Bike System Data** | The real dataset that provides trip information. |


### Data Optimization
Large datasets can be slow to process.  
Optimization means making them faster and smaller by:
- Using **Parquet files** instead of CSV (smaller and faster).  
- Converting **text columns** into **categorical columns**.  
- Keeping only useful columns and removing the rest.  

These techniques help handle millions of rows without crashing the computer.

Each tool plays a small part to make the data handling smoother and faster.






## Notion link
link : https://www.notion.so/City-Bike-Info-27ff707809b380aa9bf6eb521b19832d?source=copy_link
