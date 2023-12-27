import pymongo
from pymongo import MongoClient
import os
os.environ["COHERE_API_KEY"]='P7b3enduxOgEYHdhY4X0xBu7j7Q125sAZ3MMdSWy'
os.environ['PINECONE_API_KEY']='b31ccf3f-4c73-4851-abe6-e0653439dab5'#c85ea1ee-1d0f-4c78-9285-1dd05c5194aa'
os.environ['PINECONE_ENV']='gcp-starter'
client = MongoClient("mongodb+srv://trial:trial@trial.upeuysc.mongodb.net/?retryWrites=true&w=majority")
db=client['chatHistory']
cht_info=db.cht_info
from datetime import datetime
import pytz  # For working with time zones
print('Database file')

def insert_data(queryy,response):

# Get the current date and time
  current_datetime = datetime.now()
  ist_timezone = pytz.timezone('Asia/Kolkata')

    # Convert the UTC time to IST
  current_time_ist = current_datetime.astimezone(ist_timezone)
# Format the current date and time (optional)
  current_datetime = current_time_ist.strftime("%Y-%m-%d %H:%M:%S")
  cht_info.insert_one({'query': queryy,'response': response , 'time': current_datetime})
def get_chthistory():
  ist_timezone=pytz.timezone('Asia/Kolkata')
  cursor=cht_info.find().sort('time',pymongo.DESCENDING).limit(5)
  latest_cht_history=list()
  print('hii')
  for doc in cursor:
    qury=doc['query']
    resp=doc['response']
    tim=doc['time']
    latest_cht_history.append((qury,resp))
  #print(latest_cht_history)
  return latest_cht_history
