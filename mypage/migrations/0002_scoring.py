# Generated by Django 4.0.3 on 2022-05-12 05:44

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('mypage', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Scoring',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('nTitle', models.CharField(max_length=350)),
                ('tc', models.DecimalField(decimal_places=4, max_digits=6)),
                ('mc', models.DecimalField(decimal_places=4, max_digits=6)),
                ('gr', models.DecimalField(decimal_places=4, max_digits=6)),
                ('l', models.CharField(max_length=50)),
            ],
        ),
    ]
