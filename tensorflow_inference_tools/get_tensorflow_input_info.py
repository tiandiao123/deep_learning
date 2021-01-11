import argparse
import tensorflow as tf
from tensorflow.python.framework import graph_util
tf.reset_default_graph()  # 重置计算图

parser = argparse.ArgumentParser(description='Convert TensorFlow model into TVM')
parser.add_argument("--model_path", help="model to Convert", type=str)
args = parser.parse_args()
output_graph_path = args.model_path


# output_graph_path = 'model/model_tfnew.pb'
with tf.Session() as sess:
 
    tf.global_variables_initializer().run()
    output_graph_def = tf.GraphDef()
    # 获得默认的图
    res = []
    graph = tf.get_default_graph()
    with open(output_graph_path, "rb") as f:
        output_graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(output_graph_def, name="")
        # 得到当前图有几个操作节点
        print("%d ops in the final graph." % len(output_graph_def.node))
 
        tensor_name = [tensor.name for tensor in output_graph_def.node]
        # print(tensor_name)
        # print('---------------------------')
        # 在log_graph文件夹下生产日志文件，可以在tensorboard中可视化模型
        summaryWriter = tf.summary.FileWriter('log_graph/', graph)
 
 
        for op in graph.get_operations():
            # print出tensor的name和值
            print(op.name, op.values())
            res.append(str(op.name) + " " + str(op.values()))
    
    with open("graph.txt", "w+") as f:
        for ele in res:
            f.write(ele+"\n")
        f.close()







