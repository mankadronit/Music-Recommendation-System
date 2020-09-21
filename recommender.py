from pyspark.sql import SparkSession, SQLContext
from pyspark.mllib.recommendation import *


def evaluate_model(model, dataset):
    user_list = dataset.map(lambda x: x[0]).distinct().collect()
    artist_list = user_history.map(lambda x: x[1]).distinct()
    prediction_results = []
    for user in user_list:
        prediction_results.append(calculate_user_prediction(model, user, artist_list, dataset))
        
    avg_prediction_rate = (sum(prediction_results)/len(user_list))
    return avg_prediction_rate
    
def calculate_user_prediction(model, user, artist_list, dataset):
    train_artists = train_data.filter(lambda x: int(x[0]) == user).map(lambda x: x[1]).collect()
    
    non_train_artists = artist_list.filter(lambda x : x not in train_artists).map(lambda x : x)
    
    true_artists = dataset.filter(lambda x: int(x[0]) == user).map(lambda x: x[1])
    Y = true_artists.count()
    
    user_non_train_artists = non_train_artists.map(lambda x: (user, x))
    
    prediction_result = model.predictAll(user_non_train_artists)
    
    prediction_result = prediction_result.sortBy(lambda x: x[2], ascending = False).take(Y)
    prediction_result = sc.parallelize(prediction_result).map(lambda x: x[1])
    
    prediction_accuracy = true_artists.intersection(prediction_result)

    acc_rate = float(float(prediction_accuracy.count())/float(Y))
    return acc_rate


def map_artists(line):
    return (line[1], line[2])

def parse_user_history(line):
    return (int(line[0]), int(line[1]), line[3])

if __name__ == "__main__":
    session = SparkSession.builder.appName("Recommender").getOrCreate()
    sc = session.sparkContext
    sc.setLogLevel("ERROR")
    sqlContext = SQLContext(sc)
    data = sqlContext.read.load("s3n://music-recommendation-ronit/cleaned.csv",
                            format="com.databricks.spark.csv",
                            header=True,
                            inferSchema=True)

    df = data.toPandas()
    artist_data = data.rdd.map(map_artists).filter(lambda x: x != None).collectAsMap()
    user_history = data.rdd.map(parse_user_history)

    user_history.cache()

    train_data, validation_data, test_data = user_history.randomSplit([0.4,0.4,0.2])

    train_data.cache()
    validation_data.cache()
    test_data.cache()

    all_artists = user_history.map(lambda x: x[1]).distinct().collect()
    broadcast_var = sc.broadcast(all_artists)

    best_model = ALS.trainImplicit(train_data, rank=10)
    evaluate_model(best_model, test_data)


    recommended = best_model.recommendProducts(2, 5)

    recommendedArtists = []
    for artist in recommended:
        recommendedArtists.append(artist[1])

    artist_map = artist_data 

    # Display the results
    i = 0
    for a in recommendedArtists:
        print("Artist " + str(i) + ": " + artist_map[a])
        i = i + 1
