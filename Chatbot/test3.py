from test1 import get_entity1
import time

a, b, c, d, e, f = get_entity1()

question = input("Enter the question related to cosmetics :")
start_time = time.time()



response = f.get_response(question)
print("Response :",response)

print("Runtime : "+" %.3f Seconds " % float(time.time() - start_time))