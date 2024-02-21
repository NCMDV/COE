import json,os
from base64 import b64decode
from urllib.parse import parse_qs
import psycopg2

def lambda_handler(event, context):
    if event["httpMethod"] == "POST":
    
        DB_CONFIG = {
            "host": os.environ['host'],
            "port": os.environ['port'],
            "dbname": os.environ['dbname'],
            "user": os.environ['user'],
            "password": os.environ['password']
        }
        
        connection_uri = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}"
        
        # print(event)
        params = parse_qs(b64decode(event.get('body')).decode('utf-8'))
        ref_num = params['ref_num'][0]
        print(params)
        print(ref_num)
        
        sql_query = f"SELECT * FROM procedural_data.coe_requests_v2 where reference_id = '{ref_num}';"
        
        with psycopg2.connect(connection_uri) as conn:
            with conn.cursor() as cursor:
    
                cursor.execute(sql_query)
                result = cursor.fetchall()
                
        update_query = f"""
        UPDATE procedural_data.coe_requests_v2
        SET approval_status = 'True'
        WHERE reference_id = '{ref_num}';
        """

        if len(result) == 1:
            with psycopg2.connect(connection_uri) as conn:
                    with conn.cursor() as cursor:
                        cursor.execute(update_query)
        
            print("Edited status to True")
        else:
              print("Does not match with a single reference_id")
              
        
        message = f"received post request for {ref_num}"
        
        return {
            'statusCode': 200,
            'body': json.dumps(message),
        }

    else:
        return {
            'statusCode': 400,
            'body': json.dumps("Invalid request type"),
        }