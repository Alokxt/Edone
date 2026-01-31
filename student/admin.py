from django.contrib import admin


from student.models import *
# Register your models here.
admin.site.register(Myuser)
admin.site.register(Subject)
admin.site.register(Topic)
admin.site.register(sub_topics)
admin.site.register(roadmap)
admin.site.register(Studyparamms)
admin.site.register(all_sub_topics)
admin.site.register(all_subject)
admin.site.register(all_Topic)
admin.site.register(NewRoadmap)
admin.site.register(RoadmapStep)
admin.site.register(StudentProgress)