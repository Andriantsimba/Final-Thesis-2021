# Generated by Django 4.0.3 on 2022-05-12 05:46

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('mypage', '0002_scoring'),
    ]

    operations = [
        migrations.CreateModel(
            name='newsScoring',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('nTitle', models.CharField(max_length=350)),
                ('tc', models.DecimalField(decimal_places=4, max_digits=6)),
                ('mc', models.DecimalField(decimal_places=4, max_digits=6)),
                ('gr', models.DecimalField(decimal_places=4, max_digits=6)),
                ('l', models.CharField(max_length=50)),
            ],
        ),
        migrations.DeleteModel(
            name='Scoring',
        ),
    ]
