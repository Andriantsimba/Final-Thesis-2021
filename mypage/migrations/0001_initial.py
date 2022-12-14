# Generated by Django 4.0.3 on 2022-03-08 09:23

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Keyword',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('logasp', models.CharField(max_length=150)),
            ],
        ),
        migrations.CreateModel(
            name='News',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('news_title', models.CharField(max_length=350)),
                ('pp_title', models.CharField(max_length=250)),
                ('news_link', models.CharField(max_length=150)),
                ('news_dop', models.DateField()),
            ],
        ),
    ]
