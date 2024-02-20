import smtplib
from dotenv import load_dotenv
import os
from email.message import EmailMessage

load_dotenv()

app_password = os.getenv("app_password")
email_address = os.getenv("app_email")
receiver = "insert receiver's email here"
email_cc = "insert email here"

msg = EmailMessage()
msg['Subject'] = "Python Email Test"
msg['From'] = email_address
msg['To'] = receiver
msg['Cc'] = email_cc
msg.set_content('Sending this from python')

with smtplib.SMTP('smtp.gmail.com',587) as smtp:
    smtp.starttls()
    smtp.login(email_address,app_password)
    smtp.send_message(msg)

    print("message sent!")