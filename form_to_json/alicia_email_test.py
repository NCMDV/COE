import smtplib
from dotenv import load_dotenv
import os
from email.message import EmailMessage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

load_dotenv()

app_password = os.getenv("app_password")
email_address = os.getenv("app_email")
receiver = "insert email here"
# email_cc = "insert email here"

msg = MIMEMultipart("alternative")
msg["Subject"] = "Python HTML Email Test 2"
msg["From"] = email_address
msg["To"] = receiver
# msg['Cc'] = email_cc
reference_number = "227127COE21022024091136"

html = f"""
<html>
    <body>
        <p>Hi {receiver},<br><br>
        This is a button to approve:<br>
            
            <form action="https://063qgljmkj.execute-api.us-east-1.amazonaws.com/default/chatbot_post_receiver" method="POST">
                <button name="ref_num" value="{reference_number}">Approve</button>
            </form>
        </p>
    </body>
</html>
"""

html_content = MIMEText(html, "html")
msg.attach(html_content)

with smtplib.SMTP('smtp.gmail.com',587) as smtp:
    smtp.starttls()
    smtp.login(email_address,app_password)
    smtp.send_message(msg)

    print("message sent!")