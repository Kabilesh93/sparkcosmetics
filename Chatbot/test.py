from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer


#allocate variables for each specific products
soap = ChatBot('Soap')
perfume=ChatBot('Perfume')
lipstick=ChatBot('Lipstick')
powder=ChatBot('powder')
shampoo=ChatBot('shampoo')

#Create a new trainer for the chatbot
trainer1 = ChatterBotCorpusTrainer(soap)
trainer2 = ChatterBotCorpusTrainer(perfume)
trainer3 = ChatterBotCorpusTrainer(powder)
trainer4 = ChatterBotCorpusTrainer(lipstick)
trainer5 = ChatterBotCorpusTrainer(shampoo)



#Train the chatbot based on the soap corpus
#trainer1.train("chatterbot.corpus.Soap_domain")

#Train the chatbot based on the soap corpus
#trainer2.train("chatterbot.corpus.Perfume_domain")

#Train the chatbot based on the soap corpus
#trainer3.train("chatterbot.corpus.Powder_domain")

#Train the chatbot based on the soap corpus
#trainer4.train("chatterbot.corpus.Lipstick_domain")

#Train the chatbot based on the soap corpus
#trainer5.train("chatterbot.corpus.Shampoo_domain")


def get_entity():
        return soap, shampoo, powder, lipstick, perfume






