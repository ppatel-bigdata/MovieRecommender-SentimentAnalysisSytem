#
import re

import tweepy
from tweepy.auth import OAuthHandler
from tweepy import Stream
from tweepy.streaming import StreamListener
import socket
import json
import pandas as pd
import os

# request to get credentials at http://developer.twitter.com
consumer_key = '0oFsUmrYI59vugFr6XyqJ0C5U'
consumer_secret = 'XGeaakfoudApfwIgN6gX0Tcm83KLWlM98CwcZ8mQv5xwqTgKRY'
access_token = '2612720178-qj2HtZy96iX6zEoaUFjmpNktpOIJeOXfCGsGLEt'
access_secret = 'hTy0uckpgI9nzvB9UeSs7uGQ3VNP99p5MYyWztnJYqJJl'

# we create this class that inherits from the StreamListener in tweepy StreamListener
class TweetsListener(StreamListener):

    def __init__(self, csocket):
        self.client_socket = csocket

    # we override the on_data() function in StreamListener
    def on_data(self, data):
        try:
            message = json.loads(data)
            #print(message['text'].encode('utf-8'))
            text = message['text']
            processed_tweet = ' '.join(re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", str(text)).split())
            print(processed_tweet.encode('utf-8'))
            self.client_socket.send(message['text'].encode('utf-8'))
            return True
        except BaseException as e:
            print("Error on_data: %s" % str(e))
        return True

    def on_error(self, status):
        print(status)
        return True


def send_tweets(c_socket):
    #api = tw.API(auth)
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)
    twitter_stream = Stream(auth, TweetsListener(c_socket))
    df = pd.read_csv("movies_metadata.csv")
    df.title = df.title.astype('str')
    df = df.replace("'", '',regex=True)
    # for title in df.title:
    #    print(title)
    track_string = df["title"].to_numpy()
    track_string =['batman','Mildred Pierce','Dasepo Naughty Girls','Silentium',
                   'Ocean Heaven','Wuthering Heights']
    #print(track_string)
    twitter_stream.filter(track=track_string)

if __name__ == "__main__":
    os.chdir("D:\\trentsemester2\\bigData\\the-movies-dataset")
    new_skt = socket.socket()  # initiate a socket object
    host = "127.0.0.1"  # local machine address
    port = 5555  # specific port for your service.
    new_skt.bind((host, port))  # Binding host and port

    print("Now listening on port: %s" % str(port))

    new_skt.listen(5)  # waiting for client connection.
    c, addr = new_skt.accept()  # Establish connection with client. it returns first a socket object,c, and the address bound to the socket
    print("Received request from: " + str(addr))
    # and after accepting the connection, we can send the tweets through the socket
    send_tweets(c)
    print("end of code")