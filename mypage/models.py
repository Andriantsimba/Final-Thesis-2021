from django.db import models


# Create your models here.


class Keyword(models.Model):
    logasp = models.CharField(max_length=150)


class News(models.Model):
    news_title = models.CharField(max_length=350)
    pp_title = models.CharField(max_length=250)
    news_link = models.CharField(max_length=150)
    news_dop = models.DateField()


class newsScoring(models.Model):

    nTitle = models.CharField(max_length=350)
    tc = models.DecimalField(max_digits=6, decimal_places=4)
    mc = models.DecimalField(max_digits=6, decimal_places=4)
    gr = models.DecimalField(max_digits=6, decimal_places=4)
    l = models.CharField(max_length=50)


# class labelling(models.Model):
#     title = models.CharField(max_length=350)
#     tc = models.DecimalField(max_digits=6, decimal_places=4)
#     mc = models.DecimalField(max_digits=6, decimal_places=4)
#     gr = models.DecimalField(max_digits=6, decimal_places=4)
#     cr = models.DecimalField(max_digits=6, decimal_places=4)
#     sl = models.DecimalField(max_digits=6, decimal_places=4)
