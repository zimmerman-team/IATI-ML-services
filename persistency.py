import functools

import pymongo
MONGODB_CONN="mongodb://mongouser:XGkS1wDyb4922@localhost:27017/learning_sets"

@functools.cache
def mongo_db():
    client = pymongo.MongoClient(MONGODB_CONN)
    db = client['learning_sets']
    return db