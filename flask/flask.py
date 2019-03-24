from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from json import dumps
import urllib.parse
from urllib.request import urlopen

import app

web = Flask(__name__)
api = Api(web)

model_path = "./models"
face = app.Application(model_path)


class Employees(Resource):
    def post(self):
        data = request.get_data()
        # print(data)
        img_url = str(urllib.parse.unquote(str(data[4:])))[2:-1]

        # print(img_url)
        img_name = img_url[33:]
        # print(img_name)

        with urlopen(img_url) as response, open("./img/" + img_name, 'wb') as f_save:
            f_save.write(response.read())
            f_save.flush()
            f_save.close()
            print("成功")
        # return jsonify({'age': 91, 'gender': 'male'})
        return jsonify(face.predict("./img/" + img_name))


api.add_resource(Employees, '/employees')


if __name__ == '__main__':
    web.run(port='5001')
