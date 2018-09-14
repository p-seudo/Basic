from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
import os

bot = ChatBot("Jarvis")
bot.set_trainer(ListTrainer)
path = "/home/pseudo/Desktop/MyPy/shit-happening/chatbot/chatterbot-corpus-master/chatterbot_corpus/data/english/"
for files in os.listdir(path):
    
    data = open(path+files,'r').readlines()
    bot.train(data)
while True:
    
    message = raw_input("You :")
    if message.strip() != "Bye":
        reply = bot.get_response(message)
        print ("YourBot :",reply)
	
    if message.strip() == "Bye":
        reply = bot.get_response(message)
        print ("YourBot :",reply)
	break
