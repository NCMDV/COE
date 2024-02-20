from django.db import models
from django.utils import timezone

# Create your models here.

class Chatlogs(models.Model):
    user_message = models.CharField(blank=True, null=True)
    ai_response = models.CharField(blank=True, null=True)
    inference_time = models.FloatField(blank=True, null=True)
    total_tokens = models.FloatField(blank=True, null=True)
    prompt_tokens = models.FloatField(blank=True, null=True)
    total_cost_usd = models.FloatField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'chatlogs'


class ChatlogsTest(models.Model):
    id = models.CharField(primary_key=True)
    start_datetime = models.DateTimeField(default=timezone.now)
    duration = models.FloatField(default = 0)
    user_id = models.CharField()
    interaction_count = models.IntegerField(default = 0)
    user_typing_time = models.FloatField(default = 0)
    chatbot_inference_time = models.FloatField(default = 0)
    user_feedback = models.IntegerField(default = 0)
    escalation = models.IntegerField(default = 0)
    transcript = models.TextField(default = "")
    total_tokens = models.IntegerField(default = 0)
    prompt_tokens = models.IntegerField(default = 0)
    total_cost = models.FloatField(default = 0)

    class Meta:
        # managed = False
        db_table = 'chatlogs_test'

# class ChatFeedbacks(models.Model):
#     id = models.CharField(primary_key=True)
#     chatlog_id = models.ForeignKey(ChatlogsTest)
#     feedback = models.IntegerField(default=0)



class InsuranceData(models.Model):
    id = models.IntegerField(blank=True, null=False, primary_key=True)
    insurance_policy = models.CharField(blank=True, null=True)
    gender = models.CharField(blank=True, null=True)
    age = models.IntegerField(blank=True, null=True)
    lifestyle = models.CharField(blank=True, null=True)
    medical_history = models.CharField(blank=True, null=True)
    annual_income = models.FloatField(blank=True, null=True)
    total_premium = models.FloatField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'insurance_data'


class ItQasData(models.Model):
    question = models.CharField(blank=True, null=True)
    answer = models.CharField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'it_qas_data'


class Qna(models.Model):
    question = models.CharField(blank=True, null=True)
    answer = models.CharField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'qna'


class SharepointData(models.Model):
    file_name = models.CharField(blank=True, null=True)
    url_link = models.CharField(blank=True, null=True)
    description = models.CharField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'sharepoint_data'


class Users(models.Model):
    employee_id = models.FloatField(blank=True, null=True)
    username = models.CharField(blank=True, null=True)
    password = models.CharField(blank=True, null=True)
    email_add = models.CharField(blank=True, null=True)
    first_name = models.CharField(blank=True, null=True)
    middle_name = models.CharField(blank=True, null=True)
    last_name = models.CharField(blank=True, null=True)
    access_level = models.FloatField(blank=True, null=True)
    access_type = models.CharField(blank=True, null=True)
    department = models.CharField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'users'