# spam-filter

Spam filter application
- API using Flask
- spam classification and prediction using tensorflow library


Access the following APIs

GET: http://0.0.0.0:18080/  - Test home page

GET: http://0.0.0.0:18080/users/<user>  - Returns Hello <username>!
  
POST: http://0.0.0.0:18080/api/get_text_prediction  - needs json body in format {"text": "input string"} returns mathematical calculations from tensorflow

Presentation: https://github.com/yoivanov/spam-filter/blob/master/3_Group_Spam-E-mail.pptx
