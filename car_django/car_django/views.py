# from statistics import linear_regression
from django.http import HttpResponse
from django.template import Template, Context
from django.template.loader import get_template
from django.shortcuts import render
from django.http import JsonResponse
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from joblib import load
from django.core.files import File
import os
import numpy as np
import json
from django.core.serializers.json import DjangoJSONEncoder
from django.utils.safestring import mark_safe
from django.template import Library
import tensorflow as tf
from tensorflow.python.keras.layers import Input, Dense
from keras.models import load_model
from tensorflow import keras
from keras.models import model_from_json
from keras.models import load_model
import h5py



# import pandas as pd


def inicio(request):

    # cargo el archivo con el modelo para entrenar los datos,y descargar la ultima version
    # ----------------------------------clf-------------------------------------------------------
    clf = open(
        os.path.dirname(os.path.realpath(__file__)) + "/modelos/model_clf.pkl",
        "rb",
    )
    model_clf = load(clf)
    # ----------------------------------clf-------------------------------------------------------
    # ----------------------------------linear regresor-------------------------------------------------------
    linear_regression = open(
        os.path.dirname(os.path.realpath(__file__))
        + "/modelos/model_linear_regression.pkl",
        "rb",
    )
    model_linear_regression = load(linear_regression)
    # ----------------------------------linear regresor-------------------------------------------------------

    # ----------------------------------knn regresor-------------------------------------------------------
    knn_regressor = open(
        os.path.dirname(os.path.realpath(__file__))
        + "/modelos/model_knn_regressor.pkl",
        "rb",
    )
    model_knn_regressor = load(knn_regressor)
    # ----------------------------------knn regresor-------------------------------------------------------

    # ----------------------------------random forest regresor-------------------------------------------------------
    random_forest_regressor = open(
        os.path.dirname(os.path.realpath(__file__))
        + "/modelos/model_random_forest_regressor.pkl",
        "rb",
    )
    model_random_forest_regressor = load(random_forest_regressor)
    # ----------------------------------random forest regresor-------------------------------------------------------

    # ----------------------------------keras-------------------------------------------------------
    # keras = open(
    #     os.path.dirname(os.path.realpath(__file__)) +
    #     "/modelos/model.h5",
    #     "r",
    # )

    # with open(
    #     os.path.dirname(os.path.realpath(__file__)) + "/json/model_keras.json",
    #     "r",
    # ) as f:
    #     model_keras = model_from_json(f.read())
    # # load model weights
    # model_kerasssss = model_keras.load_weights(os.path.dirname(os.path.realpath(__file__)) + "/modelos/model.h5","r")

    # model_keras = keras.models.load_model("./my_model_definitivo_keras.h5")

    # ----------------------------------keras-------------------------------------------------------

    marcas_id = open(
        os.path.dirname(os.path.realpath(__file__)) + "/json/marcas_id.json",
        "rb",
    )

    marca_model_id = open(
        os.path.dirname(os.path.realpath(__file__)) +
        "/json/marca_model_id.json",
        "rb",
    )

    with marcas_id as file:
        data = json.load(file)
        data_json = json.dumps(data)

    with marca_model_id as file2:
        data2 = json.load(file2)
        data_json2 = json.dumps(data2)

    if request.POST:
        # fuelTypeId	km	makeId	modelId	transmissionTypeId	year	cubicCapacity	doors	hp
        fuelTypeId = request.POST["fuelTypeId"]
        km = request.POST["km"]
        makeId = request.POST["makeId"]
        modelId = request.POST["modelId"]
        transmissionTypeId = request.POST["transmissionTypeId"]
        year = request.POST["year"]
        cubicCapacity = request.POST["cubicCapacity"]
        doors = request.POST["doors"]
        hp = request.POST["hp"]

        data_usuario = np.array(
            [
                [
                    fuelTypeId,
                    km,
                    makeId,
                    modelId,
                    transmissionTypeId,
                    year,
                    cubicCapacity,
                    doors,
                    hp,
                ]
            ]
        )

        modelos_training = [model_linear_regression, model_knn_regressor,
                            model_random_forest_regressor,model_clf]
        predicts = []

        for modelo in modelos_training:
            predict = modelo.predict(data_usuario)
            predict = float(predict)
            predicts.append(predict)

        # predict_keras = model_keras.predict(data_usuario)
        # print(predict_keras)

        res = dict(zip(modelos_training, predicts))

        print("diccttt:", res)
        # print("post:", request.POST)
        return render(
            request,
            "resultado.html",
            {"predicts": predicts, "models": modelos_training, "res": res},
        )

    # print("data json",data)
    return render(
        request, "inicio.html", {"marcas_id": data_json, "marca_model_id": data_json2}
    )


def resultado(request):
    print(request.POST)
    if request.POST:
        data = {"dct": request.POST}
        print(f"DATA:{data}")
        return render(request, "resultado.html")
