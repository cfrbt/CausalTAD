# CausalTAD

The tested python version is 3.9.13, and please refer to "requirements.txt" for package versions.

All datasets, including trajectories and information of road network, can be found in the directory "datasets".

The code can be found in the directory "code". You can modify the hyperparamaters by editing the file "code/CausalTAD/params.py", and then train the model and evaluate it by switching to the directory "code" and running:
~~~
python main.py
~~~
During training, the parameters of the model will be saved per epoch, and you can choose a saved model to load by editing the parameter "load_model" in "code/main.py". Finally, run following line to calculate the metrics:
~~~
python evaluate.py
~~~
