#!/usr/bin/env bash

python3 mnist_keras_at.py --save_model True --model_type a --origin_method fgsm
python3 mnist_keras_at.py --save_model True --model_type a --origin_method bim
python3 mnist_keras_at.py --save_model True --model_type a --origin_method mifgsm

python3 mnist_keras_at.py --save_model True --model_type b --origin_method fgsm
python3 mnist_keras_at.py --save_model True --model_type b --origin_method bim
python3 mnist_keras_at.py --save_model True --model_type b --origin_method mifgsm

python3 mnist_keras_at.py --save_model True --model_type c --origin_method fgsm
python3 mnist_keras_at.py --save_model True --model_type c --origin_method bim
python3 mnist_keras_at.py --save_model True --model_type c --origin_method mifgsm