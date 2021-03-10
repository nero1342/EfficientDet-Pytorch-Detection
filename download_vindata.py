token = {"username":"nero18","key":"d8628b574fd3cc95931821e3399a914a"}
import json 
with open('kaggle.json','w') as token_file:
  json.dump(token, token_file)