

def extract_entity1(str):
    if str.find("soap" or"soaps") != -1:
        return "soap"
    else:
        if str.find("perfume" or "citrus" or "fruity" or "oriental" or "spicy" or "perfumes") != -1 :
            return "perfume"
        else:
            if str.find("powder" or "powders"or "ponds") != -1:
                return "powder"
            else:
                if str.find("shampoo" or "shampoos" or "heand and shoulder" or "sunsilk") != -1:
                    return "shampoo"
                else:
                    if str.find("lipstick" or "lipsticks" or "lips" or "nyx") != -1:
                        return "shampoo"
                    else:
                        return "general"


# entity=extract_entity1("what are the brands of shampoo in your market")
# print(entity)
