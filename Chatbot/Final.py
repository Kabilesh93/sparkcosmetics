
from test1 import get_entity1
from flask import Flask,request,jsonify
from flask_cors import CORS,cross_origin
from test22 import extract_entity1

app = Flask(__name__)
CORS(app)

@app.route("/chat", methods=["POST","GET"])
def get_response():
    str=request.get_json()
    response=''
    question = str.get("message","none")


    a, b, c, d, e, f, g = get_entity1()
    entity = extract_entity1(question)
    #print("entity is:",entity)

    if (entity == "soap"):
        response = a.get_response(question).text

    else:
        if (entity == "shampoo"):
            response = b.get_response(question).text

        else:
            if (entity == "powder"):
                response = c.get_response(question).text

            else:
                if (entity == "lipstick"):
                    response = d.get_response(question).text

                else:
                    if (entity == "perfume"):
                        response = e.get_response(question).text
                    else:
                        if (entity == "general"):
                            #print(entity)
                            response = g.get_response(question).text
                        else:
                            response="Sorry My questions are Limited to Cosmetic Domain"
#reinforcement learning
    with open(entity + '.yml') as myfile:
        if not ("- - " + question) in myfile.read():
            file = open('updated.yml', "a")
            file.writelines("- - " + question + "\n")
            file.writelines("  - " + response + "\n")
            file.close()
    # print(response.text)

    return jsonify(response)



if __name__ == "__main__":
   app.run(host='127.0.0.1', port=5003, debug=True)


# question = input("Enter the question related to cosmetics :")
# print(get_response(question))