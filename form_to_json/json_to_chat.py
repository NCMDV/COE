import json, os
from datetime import datetime

# SPECIFY FILE
f = open(os.path.join('./json_files','form.json'))
file = f.read()
data = json.loads(file)

# Print title of form
form_title = data["metadata"]["title"]
print(form_title.upper())

# Initialize variables for answer
json_answer = {"title":form_title}
q_and_a = []

# Iterating over the form elements
elements = data["items"]
for e in elements:
    q_type = e['type']
    title = e["title"]
    # LIST is a dropdown
    if q_type == "LIST" or q_type =="MULTIPLE_CHOICE":
        print(f"\n{title}")
        choices = e["choices"]
        choices_lower = [x.lower() for x in choices]
        print(f"choices: {choices}")
        choice = None
        while choice is None:
             user_answer = input("please type your desired choice: ")
             if user_answer.lower() in choices_lower:
                 choice = user_answer.lower()
             elif user_answer.lower() == 'exit':
                break
             else:
                print("{input} is not in the given choices".format(input=user_answer))
                continue
        q_and_a.append({"question":title,"answer":choice})    

    elif q_type == "CHECKBOX":
        print(f"\n{title}")
        choices = e["choices"]
        choices_lower = [x.lower() for x in choices]
        print(f"choices: {choices}")
        choice_list = []
        list_match = False
        while not list_match:
            user_answer = input("please type your desired choice/s separated by comma: ")
            answer_list = user_answer.split(",")
            answer_list_lower = [x.lower().strip() for x in answer_list]
            diff = set(answer_list_lower).difference(set(choices_lower))
            if user_answer.lower() == 'exit':
                break
            elif len(diff) == 0:
                list_match = True
                q_and_a.append({"question":title,"answer":answer_list_lower})
            else:
                print(f"{diff} not in choices")

    elif q_type == "DATE":
        # specify date format
        # sample format: "04-01-1995", dd-mm-yyyy
        print(f"\n{title}")
        format = "%d-%m-%Y"
        date_match = False
        while not date_match:
            user_answer = input("please type date in 'dd-mm-yyyy' format: ")
            answer = user_answer.strip()
            if answer.lower() == 'exit':
                break
            try:
                if bool(datetime.strptime(answer, format)):
                    q_and_a.append({"question":title,"answer":answer})
                    date_match = True
            except:
                print("{input} is not in the valid format".format(input=user_answer))

    elif q_type == "PAGE_BREAK":
        print(f"\n{title}")
        print(e["helpText"])

    elif q_type == "IMAGE":
        print("\nImage type, not yet implemented")
        print(title)

    elif q_type == "TEXT":
        text_val = input(f"\n{title}: ").strip()
        q_and_a.append({"question":title,"answer":text_val})

    else:
        print("\nOther type, not yet implemented")

json_answer["items"] = q_and_a
print("------")
print(json_answer)