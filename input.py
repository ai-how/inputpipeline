import tensorflow as tf

##================================================##
## good practice to limit data processing to CPU  ##
##================================================##
def return_batch(train, list_of_columns):
  with tf.device("/cpu:0"):
    train = train.sample(frac=1).reset_index(drop=True) # ensure the train data is shuffled at each epoch
    X_train = train[[list_of_columns]] # list_of_columns indicating the set of features to be used for training
    Y_train = train['labels'] # labels corresponds to target variable
    return X_train, Y_train
  
queue_capacity = 2000 # indicates how much a queue can hold at any time

queue = tf.RandomShuffleQueue(shapes = [[no_of_features],[]],
                                       dtypes = [tf.float32, tf.int32
                                       capacity=queue_capacity
                                       min_after_dequeue=1000)
                                      
##================================================##
## placeholder to hold features and labels data   ##
##================================================##
X = tf.placeholder(dtype=tf.float32, shape = [None,no_of_features])
Y = tf.placeholder(dtype=tf.int32, shape = [None,])

##================================================##
## operation to fill and close the queue          ##
##================================================##
enqueue_op = queue.enqueue_many([X,Y])
close_op = queue.close()

##================================================##
## operation to fetch mini-batches in training    ##
##================================================##
X_batch, Y_batch = queue.dequeue_many(128) # 128 is the size of mini-batch, ususally a hyperparameter

def enqueue(sess, train,list_of_columns):
  for i in range(no_epochs): # run the loop for no. of epochs used for model training
    X_train, Y_train = return_batch(train, list_of_columns)
    start_pos =0
    ##================================================##
    ## ensures the queue is filled all the time       ##
    ##================================================##
    while start_pos < X_train.shape[0]:
      end_pos = start_pos+queue_capacity 
      feed_X = X_train[start_pos:end_pos]
      feed_Y = Y_train[start_pos:end_pos]
      sess.run(enqueue_op, feed_dict = {X: feed_X, Y: feed_Y})
      start_pos + = queue_capacity
     sess.run(close_op)
     
##================================================##
## operation to start the queue and fetch data    ##
##================================================##
    
with tf.Session as sess:
  tf.train.start_queue_runners(sess=sess)
  enqueue_thread = threading.Thread(target=enqueue, args=(sess,train,list_of_columns))
  enqueue_thread.start()
  ##================================================##
  ## fetch minibatches in each iteration            ##
  ##================================================##
  for i in range(no_epochs):
    for in range(no_of_iter):
      batch_X, batch_Y = sess.run([X_batch, Y_batch])
      # batch_X and batch_y shapes are (128,no_of_features), (1,128)
      # use these mini-batches to train your AI models
