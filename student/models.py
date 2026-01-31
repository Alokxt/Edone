from django.db import models


from django.contrib.auth.models import User
from django.utils import timezone
class all_subject(models.Model):
    
    name = models.CharField(max_length=250)
    def __str__(self):
        return self.name
class all_Topic(models.Model):
    
    name = models.CharField(max_length=30)
    subject = models.ForeignKey(all_subject,on_delete=models.CASCADE,related_name="alltopics")

    def __str__(self):
        return self.name
    
class all_sub_topics(models.Model):
   
    
    name=models.CharField(max_length=250)
   
    topic = models.ForeignKey(all_Topic,on_delete=models.CASCADE,related_name="allsubtopics")

    def __str__(self):
        return self.name

class Myuser(models.Model):
    email =  models.EmailField(blank=True,null=True,default=None)
    student = models.ForeignKey(User,default=None,on_delete=models.CASCADE)
    profile = models.ImageField(blank=True,upload_to='media/')
    last_iq_quiz = models.DateTimeField(null=True,blank=True)
    is_mail_verified = models.BooleanField(default=False)


    def needs_quiz(self):
        if not self.last_iq_quiz:
            return True 
        delta = timezone.now() - self.last_iq_quiz
        return delta.days >= 7

    
class CompExams(models.Model):
    choices = [
        ('GATE', 'GATE'),
        ('UPSC', 'UPSC'),
        ('SSC', 'SSC'),
        ('BANKING', 'BANKING'),
    ]
    name = models.CharField(choices=choices,unique=True,max_length=20)

    def __str__(self):
        return self.name 
class ExamNews(models.Model):
    exam = models.ForeignKey(CompExams,on_delete=models.CASCADE,related_name='news')
    title = models.CharField(max_length=200)
    description = models.TextField(blank=True,null=True)
    source = models.URLField(blank=True,null=True)
    published_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title
    

class NewsMedia(models.Model):
    news = models.ForeignKey(ExamNews,on_delete=models.CASCADE,related_name='media')
    file = models.FileField(upload_to='media/')
    tag = models.CharField(max_length=400)

    def __str__(self):
        return self.tag 


class UserPrepration(models.Model):
    student = models.ForeignKey(Myuser,on_delete=models.CASCADE,related_name='preprations')
    exam = models.ForeignKey(CompExams,on_delete=models.CASCADE,related_name='students')
    class Meta:
        unique_together = ('student', 'exam')


class Subject(models.Model):
    student = models.ForeignKey(User,default=None,on_delete=models.CASCADE,related_name='subjects',null=True)
    sub = models.ForeignKey(all_subject,on_delete=models.DO_NOTHING,related_name='studentsub',blank=True,null=True)
    name = models.CharField(max_length=250,null=True,blank=True)
    def __str__(self):
        return self.name
    
class Topic(models.Model):
    student = models.ForeignKey(User,default=None,on_delete=models.CASCADE,related_name='students_topics',null=True)
    top = models.ForeignKey(all_Topic,on_delete=models.DO_NOTHING,related_name='totopic',null=True,blank=True)

    name = models.CharField(max_length=30,null=True,blank=True)
    subs = models.ForeignKey(Subject,on_delete=models.CASCADE,related_name="topics")

    @property
    def completed_percentage(self):
        subs = self.subtopics.all()
        total = subs.count()
        if total == 0:
            return 0
        completed = subs.filter(is_completed=True).count()
        return (completed / total) * 100
    def __str__(self):
        return self.name

    
class sub_topics(models.Model):
    student = models.ForeignKey(User,on_delete=models.CASCADE,related_name='students_subtopics',null=True,blank=True)
    subt = models.ForeignKey(all_sub_topics,on_delete=models.DO_NOTHING,related_name='tosubtopics',null=True,blank=True)
    name=models.CharField(max_length=250,null=True,blank=True)
    is_completed = models.BooleanField(default=False)
    topic = models.ForeignKey(Topic,on_delete=models.CASCADE,related_name="subtopics")

    def __str__(self):
        return self.name
    


class roadmap(models.Model):
    register = models.BooleanField(default=False)
    student = models.ForeignKey(User,default=None,on_delete=models.CASCADE,related_name='studentroadmap',null=True,blank=True)
    name = models.CharField(max_length=250)
    subjects = models.ManyToManyField(Subject,related_name='subjects',blank=True)
    st_sub = models.ManyToManyField(all_subject,related_name='defaultmap',blank=True)
    start_date = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
   

    def __str__(self):
        return self.name
    

class Studyparamms(models.Model):
    student = models.ForeignKey(User,on_delete=models.CASCADE,related_name='predictperformance')
    study_hours = models.DecimalField(max_digits=10,decimal_places=2,default=2.0)
    days = models.IntegerField(default=0)
    streak = models.IntegerField(default=0)
    past_exam_performance = models.DecimalField(max_digits=5,decimal_places=2,default=None)
    iq_score = models.DecimalField(max_digits=3,decimal_places=1,default=5.0)

    @property
    def average_study_hours(self):
        return self.study_hours/self.days
    

class iq_test(models.Model):
    student = models.ForeignKey(Myuser,on_delete=models.CASCADE,related_name='studentiqtest')
    score = models.DecimalField(default=0.0,max_digits=5,decimal_places=2)
    date  = models.DateTimeField(null=True,blank=True)
    test_data = models.TextField(blank=True)
    clear = models.BooleanField(default=False)


class NewRoadmap(models.Model):
    name = models.CharField(max_length=100)
    description = models.TextField(blank=True)
    created_by = models.ForeignKey(User, on_delete=models.CASCADE, related_name="roadmaps",blank=True,default=None,null=True)
    is_official = models.BooleanField(default=False)  
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name
    
    @property
    def completion_percentage(self):
        total = self.steps.count()
        if total == 0:
            return 0
        completed = StudentProgress.objects.filter(
            student=self.created_by, step__roadmap=self, completed=True
        ).count()
        return round((completed / total) * 100, 2)

class RoadmapStep(models.Model):
    roadmap = models.ForeignKey(NewRoadmap, on_delete=models.CASCADE, related_name="steps")
    title = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    order = models.PositiveIntegerField(default=0)  

    def __str__(self):
        return f"{self.roadmap.name} - {self.title}"
    
class StudentProgress(models.Model):
    student = models.ForeignKey(User, on_delete=models.CASCADE, related_name="progress")
    step = models.ForeignKey(RoadmapStep, on_delete=models.CASCADE, related_name="progress")
    completed = models.BooleanField(default=False)
    completed_on = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        return f"{self.student.username} - {self.step.title} ({'Done' if self.completed else 'Pending'})"