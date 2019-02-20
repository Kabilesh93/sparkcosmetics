from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer


#allocate variables for each specific products
soap = ChatBot('Soap')
perfume=ChatBot('Perfume')
lipstick=ChatBot('Lipstick')
powder=ChatBot('powder')
shampoo=ChatBot('shampoo')
general=ChatBot('general')

collective=ChatBot('collective')

#Create a new trainer for the chatbot
trainer1 = ChatterBotCorpusTrainer(soap)
trainer2 = ChatterBotCorpusTrainer(perfume)
trainer3 = ChatterBotCorpusTrainer(powder)
trainer4 = ChatterBotCorpusTrainer(lipstick)
trainer5 = ChatterBotCorpusTrainer(shampoo)

trainer6 = ChatterBotCorpusTrainer(collective)
trainer7 = ChatterBotCorpusTrainer(general)


#train the chatbot based on the soap corpus
# trainer1.train('C:/Users/Suganthan/PycharmProjects/Decoders/final_year/Chatbot/data/Soap_domain')
# #
# # Train the chatbot based on the perfume corpus
# trainer2.train('C:/Users/Suganthan/PycharmProjects/Decoders/final_year/Chatbot/data/Perfume_domain')
# #
# # Train the chatbot based on the powder corpus
# trainer3.train('C:/Users/Suganthan/PycharmProjects/Decoders/final_year/Chatbot/data/Powder_domain')
# #
# # Train the chatbot based on the lipstick corpus
# trainer4.train('C:/Users/Suganthan/PycharmProjects/Decoders/final_year/Chatbot/data/Lipstick_domain')
# #
# # Train the chatbot based on the shampoo corpus
# trainer5.train('C:/Users/Suganthan/PycharmProjects/Decoders/final_year/Chatbot/data/Shampoo_domain')
# #
# # Train the chatbot based on the collective corpus
# trainer6.train('C:/Users/Suganthan/PycharmProjects/Decoders/final_year/Chatbot/data/collective')
# #
# # Train the chatbot based on the general corpus
# trainer7.train('C:/Users/Suganthan/PycharmProjects/Decoders/final_year/Chatbot/data/general')

def get_entity1():
        return soap, shampoo, powder, lipstick, perfume, collective, general
