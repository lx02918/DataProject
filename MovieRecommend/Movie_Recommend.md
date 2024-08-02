# Movie Recommendation System Based on Collaborative Filtering Algorithm

**Project Introduction**

This project involves creating a movie recommendation system based on partial movie data. It enables basic functions such as registration, login, rating, and recommendation. The recommendation is achieved using a collaborative filtering algorithm and user information and ratings are stored in a MySQL database.

**Tech Stack**

Python, Django, MySQL, jQuery, CSS, HTML

**Project Highlights**

+ Use the UserCF algorithm to simulate user-movie matrix data to recommend movies to users based on other users with similar interests.
+ Use the ItemCF algorithm to recommend movies similar to those the user has liked before.
+ Complete the connection between the front-end and back-end using the Django framework and gradually improve the functionality.
  
**Project Achievements**

1. Able to recommend movies to users based on existing movie data.
2. A web page that allows interactive user registration, login, rating of watched movies, submitting ratings, and viewing recommended movie lists.

## Issues Encountered in the Project

1. Django
    Before starting the project, I had not used Django. After some research, I decided to use it to build the web structure. However, I quickly encountered the first problem: frequent errors in local development. Later, I found that local development might cause conflicts due to other installed libraries. I resolved this by using a virtual environment for development. This was something I hadn’t anticipated as I usually use the Conda environment and forgot about potential local conflicts.

    Writing front-end code in Django isn’t straightforward. Initially, with some help from AI and debugging, I completed a simple page for user login and the homepage. However, I couldn’t open it after completion. Later, I found that slight modifications to the HTML and CSS were needed, which resolved the issue and allowed the page to function properly.

2. Data
    Initially, I only had movie rating data. Later, I found a dataset containing movie poster links for aesthetics. The dataset needed merging due to multiple CSV files. The merging process initially failed due to special characters in movie names and links. I eventually found that the enclosed by statement was the cause.

    I chose Navicat for database connection. Initially, I encountered a 1209 error during data import. The solution involved using a specified folder for import, which I initially thought was a MySQL issue. After placing files in the specified folder, the problem persisted. It turned out to be a Navicat compatibility issue, resolved by using compatible versions of both tools. This issue arose from my inexperience with connection tools, having always used MySQL directly.

3. Web Page
 I had no prior experience with web development. Even completing a visualization page with Echarts was challenging. The web page development for this project was mostly assisted by AI. Although projects should ideally be completed independently, I believe AI can be a useful tool. It significantly helped me complete the programming tasks.


## Unresolved Issues

1. Refresh Problem
    Due to my lack of front-end experience, page refreshes result in duplicate values in the database and regenerate the recommendation list each time. I haven't found a solution yet.

2. Recommendation Algorithm
    UserCF is relatively faster as it recommends based on common user interests. ItemCF, which recommends movies similar to those previously liked, is slower due to the large number of movies. I’m considering if storing data in Hive and using other tools might improve performance.

3. Layout
    This is a significant issue for which I currently have no good solution. I plan to study it further to resolve it later.