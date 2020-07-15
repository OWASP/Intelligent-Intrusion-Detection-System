#Serializers for input
from rest_framework import serializers
from rest_framework.renderers import JSONRenderer

class EndpointSerializer(serializers.ModelSerializer):
    read_only_fields = ("model_name", "param1", "param2", "param3")


serializer = EndpointSerializer(data)
json = JSONRenderer().render(serializer.data)

#Deserializinf of data
from StringIO import StringIO
from rest_framework.parsers import JSONParser

stream = StringIO(json)
data = JSONParser().parse(stream)

