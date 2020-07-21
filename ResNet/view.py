from tensorflow.python.tools.import_pb_to_tensorboard import import_to_tensorboard
import os

#model_dir = 'model'
#model = os.path.join(model_dir, 'saved_model1.pb')
model = os.path.join('resnet50_v1.pb')
print(model)
import_to_tensorboard(model_dir=model, log_dir='log', tag_set='fc1000')
