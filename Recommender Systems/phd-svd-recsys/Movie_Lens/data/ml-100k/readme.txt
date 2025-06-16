🎯 Scenario 1: Adding 40 New Users (5 Users per Decade)

- Users 944 to 948:  
   - Favorite decade = 1920 
   - Number of movies in 1920 = 2
   - Each of these 5 users rates 2 movies:
   5 * 2 = 10
   
---

- Users 949 to 953:  
   - Favorite decade = 1930
   - Number of movies in 1930 = 29
   5 * 29 = 145
---

- Users 954 to 958:  
   - Favorite decade = 1940 
   - Number of movies in 1940 = 45  
   5 * 45 = 225 
---

- Users 959 to 963:  
   - Favorite decade = 1950 
   - Number of movies in 1950 = 54  
    5 * 54 = 270 
---

- Users 964 to 968:  
   - Favorite decade = 1960 
   - Number of movies in 1960 = 43  
    5 * 43 = 215
---

- Users 969 to 973 :  
   - Favorite decade =   1970   
   - Number of movies in 1970 =   53   
      
   5  * 53 = 265    new ratings.  
    
---

- Users 974 to 978 :  
   - Favorite decade =   1980   
   - Number of movies in 1980 =   107   
      
   5  * 107 = 535    new ratings.  
    
---

- Users 979 to 983 :  
   - Favorite decade =   1990   
   - Number of movies in 1990 =   1348   
      
   5  * 1348 = 6740    new ratings.  
    
---

### 📊   Total New Ratings After Adding 40 Users:  

10 + 145 + 225 + 270 + 215 + 265 + 535 + 6740 = 8405


✅   Final Dataset Size:  

Dataset Size After 40 Users   = 100,000 + 8405 = 108,405

---

## 🎯  Scenario 2: Adding 80 New Users (10 Users per Decade)  

- Same logic, but now   10 users per decade.  
- Multiply the number of movies by 10:

   
  Total New Ratings for 80 Users   = 2  * 10 + 29  * 10 + 45  * 10 + 54  * 10 + 43  * 10 + 53  * 10 + 107  * 10 + 1348  * 10


   
  Total New Ratings for 80 Users   = 16810
 

✅   Final Dataset Size:  

   
  Dataset Size After 80 Users   = 100,000 + 16,810 = 116,810
 

---

## 🎯   Scenario 3: Adding 120 New Users (15 Users per Decade)  

- Now   15 users per decade.  
- Multiply the number of movies by 15:

Total New Ratings for 120 Users = 2 * 15 + 29 * 15 + 45 \imes 15 + 54 * 15 + 43  * 15 + 53  * 15 + 107  * 15 + 1348  * 15
 

   
  Total New Ratings for 120 Users   = 25,215
 

✅   Final Dataset Size:  

   
  Dataset Size After 120 Users   = 100,000 + 25,215 = 125,215
 

---

🔥   Summary of New Ratings Added:  

| Scenario | New Users  | New Ratings Added| Final Dataset Size |
|----------|------------|------------------|--------------------|
| 40 Users | 40         | 8,405            | 108,405            |
| 80 Users | 80         | 16,810           | 116,810            |
| 120 Users| 120        | 25,215           | 125,215            |
