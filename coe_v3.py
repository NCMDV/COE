import psycopg2
from datetime import datetime
from db_creds import *

DB_CONFIG = {
    "host": host,
    "port": port,
    "dbname": dbname,
    "user": user,
    "password": password
}

connection_uri = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}"



is_filing = True
while is_filing:
    #stores the progress data inputted by user for coe filing
    coe_filing_data = {}

    travel_synonyms = ["travel", "abroad", "flying", "visiting abroad", "visit", "vacation"]
    credit_card_synonyms = ["credit card", "loan", "credit", "card"]

    purpose_stated = False
    while not purpose_stated:

        purpose = input("What is the purpose of your COE filing?")
        
        if any(keyword in purpose.lower() for keyword in travel_synonyms):
            purpose, coe_filing_data['purpose'] = "Travel", "Travel" 
            purpose_stated = True
        
        elif any(keyword in purpose.lower() for keyword in credit_card_synonyms):
            purpose = "Credit Card"
            coe_filing_data['purpose'] = "Credit Card Application"
            purpose_stated = True
        
        else:
            print("I didn't quite get that. Is it for travelling or Credit Card Application?")
            purpose_stated = False
        
        #create sql query based on purpose
        sql_query = f"SELECT * FROM procedural_data.coe_questions_v2 where purpose = '{purpose}';"


    with psycopg2.connect(connection_uri) as conn:
        with conn.cursor() as cursor:

            cursor.execute(sql_query)
            results = cursor.fetchall()

    #ai iterates on the question and user provides input
    for q_id, purpose, question, header,data_type in results:
        user_answer = input(question)
        coe_filing_data[header] = user_answer
        

    #prints the summary of the filing
    print("*"*100)
    print("Here is the summary of your Filing. Kindly check if the information are correct:")
    print("*"*100)
    for key, value in coe_filing_data.items():
        print(f"{key.capitalize()} : {value}")
    print("*"*100)

    #after summary is displayed, checks if the user wants to proceed, modify and cancel
    is_proceed = True
    while is_proceed:
        response = input("Would you like to proceed on your COE filing?")

        if "yes" in response.lower():
            is_proceed = False  #done with proceed
            is_filing = False   #done with filing

            #save in db
            create_table_query = """
            CREATE TABLE IF NOT EXISTS procedural_data.coe_requests_v2 (
            emp_id VARCHAR(255),
            purpose VARCHAR(255),
            details TEXT,
            request_date DATE,
            request_time TIME);"""


            with psycopg2.connect(connection_uri) as conn:
                with conn.cursor() as cursor:
                    cursor.execute(create_table_query)
                    purpose = coe_filing_data["purpose"]
                    del coe_filing_data["purpose"]
                    # sql_query = "INSERT INTO procedural_data.coe_requests (emp_id, purpose, details, request_date, request_time) VALUES (%s, %s, %s, %s, %s);"
                    sql_query = "INSERT INTO procedural_data.coe_requests_v2 (reference_id, emp_id, purpose, details, request_date, request_time,approval_status) VALUES (%s, %s, %s, %s, %s, %s, %s);"
                    emp_id = coe_filing_data["emp_id"]
                     

                    # IDNUMBER + FIRST LETTERS OF TITLE OF FORM + DATE IN DDMMYYYY + TIMEREQUEST HHMMSS
                    # 23-2547 + Certificate of Employment + 20022024 + 140059
                    # 232547COE20022024140059
                    reference_id = emp_id.replace("-", "") + "COE" + datetime.now().strftime("%d%m%Y%H%M%S")

                    sql_params = [reference_id,
                                  emp_id, 
                                  purpose, 
                                  str(coe_filing_data), 
                                  datetime.now().date(), 
                                  datetime.now().time(),
                                  "Pending"
                                  ]
                    cursor.execute(sql_query, sql_params)
            print("Application is already saved in DB.")


        elif any(keyword in response.lower() for keyword in ["no", "exit"]):
            #back to intent classification
            is_proceed = False      #no longer wants to proceed
            is_filing = False        #no longer wants to file

        elif any (keyword in response.lower() for keyword in ["edit", "modify", "change"]):
            #allows user to modify the coe filing hence, asks again for the input
            is_proceed = False

        else:
            print("I didn't quite get that.")



print("Exit na sa COE request intent.")
