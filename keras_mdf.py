import numpy as np
from tensorflow.keras.models import load_model
from modeci_mdf.execution_engine import EvaluableGraph
from modeci_mdf.utils import simple_connect
from modeci_mdf.mdf import *

#function to load keras model
def keras_model():
    model= load_model('model.h5')
    return model

#function to load weight and activation for the model layers cudart64_110.dll
def weight_and_activation(model):
    params= {}
    activation=[]
    layers_of_model= ['dense', 'dense_1', 'dense_2', 'dense_3', 'dense_4']
    for item in layers_of_model:
        dic= {}
        layer= model.get_layer(item)
        weight,bias = layer.weights
        dic['weights']= np.array(weight)
        dic['bias']= np.array(bias)
        params[item]= dic
        activation.append(str(layer.activation).split()[1])
    return params, activation


params, activation= weight_and_activation(model= keras_model())

# function to convert the keras to Mdf
def converting_keras_mdf(input_array):
    global activate_1, activate_2, activate_3, activate_4, activate_5
    # creating model and appending to graph
    model_mdf= Model(id= 'AMELA')
    graph= Graph(id= 'Converting Keras to MDF')
    model_mdf.graphs.append(graph)

    # create input layer node

    input_layer= Node(id= 'input layer')
    input_layer.parameters.append(Parameter(id= 'input_layer', value=input_array))
    input_layer.output_ports.append(OutputPort(id= 'output', value= 'input_layer'))
    graph.nodes.append(input_layer)


    # create first layer node, append to graph and connect it to input node
    first_node= Node(id='dense_layer', metadata= {'color':'.8 .8 .8'})
    first_node.input_ports.append(InputPort(id= 'input'))
    first_node.parameters.append(Parameter(id='dense_weight', value= params['dense']['weights']))
    first_node.parameters.append(Parameter(id='dense_bias', value= params['dense']['bias']))
    feedForward= Parameter(id= 'feedForward',value= '(dense_weight) + dense_bias')
    first_node.parameters.append(feedForward)
    first_node.output_ports.append(OutputPort(id= 'dense_output', value= 'feedForward trained network'))
    graph.nodes.append(first_node)
    simple_connect(input_layer, first_node, graph)

    # creating activation for the first node and,appending to the graph and connect it to input layer node
    activate_1= Node(id= 'dense_activation')
    activate_1.input_ports.append(InputPort(id='input'))
    activation= Parameter(id= 'relu',value = 'input*input')
    
    activation.conditions.append(ParameterCondition(id= 'test',
                            test= 'relu=0', value = 'relu'))

    activate_1.parameters.append(activation)
    activate_1.output_ports.append(OutputPort(id='activate_1', value= 'relu' ))
    graph.nodes.append(activate_1)
    simple_connect(first_node, activate_1, graph)
    
    
    # create second layer node, append to graph and connect it to first layer activation node
    second_node= Node(id='dense1_layer', metadata= {'color':'.8 .8 .8'})
    second_node.input_ports.append(InputPort(id= 'input'))
    second_node.parameters.append(Parameter(id='dense1_weight', value= params['dense_1']['weights']))
    second_node.parameters.append(Parameter(id='dense1_bias', value= params['dense_1']['bias']))
    feedForward= Parameter(id= 'feedForward',
                           value= '(dense1_weight) + dense1_bias')
    second_node.parameters.append(feedForward)

    second_node.output_ports.append(OutputPort(id= 'dense1_output', value= 'feedForward'))

    graph.nodes.append(second_node)

    simple_connect(activate_1, second_node, graph)


    # creating second layer activation node, to append to graph and connect it to second layer node
    activate_2= Node(id= 'dense1_activation')
    activate_2.input_ports.append(InputPort(id='input'))
    activation= Parameter(id= 'relu',
                      value = 'input*input')
    activation.conditions.append(ParameterCondition(id= 'test',
                            test= 'relu=1', value = 'relu'))

    activate_2.parameters.append(activation)
    activate_2.output_ports.append(OutputPort(id='activate_2', value= 'relu' ))
    graph.nodes.append(activate_2)
    simple_connect(second_node,activate_2, graph)

    # create third layer node, append to graph and connect it to second layer activation node

    third_node= Node(id='dense2_layer', metadata= {'color':'.8 .8 .8'})
    third_node.input_ports.append(InputPort(id= 'input'))
    third_node.parameters.append(Parameter(id='dense2_weight', value= params['dense_2']['weights']))
    third_node.parameters.append(Parameter(id='dense2_bias', value= params['dense_2']['bias']))
    feedForward= Parameter(id= 'feedForward',
                           value= '(dense2_weight) + dense2_bias')
    third_node.parameters.append(feedForward)

    third_node.output_ports.append(OutputPort(id= 'dense2_output', value= 'feedForward'))
    graph.nodes.append(third_node)
    simple_connect(activate_2, third_node, graph)
    
    # create third layer activation node, append to graph and connect it to third layer node

    activate_3= Node(id= 'dense2_activation')
    activate_3.input_ports.append(InputPort(id='input'))
    activation= Parameter(id= 'sigmoid', 
                         value= '1/(1 + (2.71828**(-input)))' )
    activate_3.parameters.append(activation)
    activate_3.output_ports.append(OutputPort(id='activate_2', value= 'sigmoid' ))
    graph.nodes.append(activate_3)
    simple_connect(third_node, activate_3, graph)
    
    # create fourth layer node, append to graph and connect it to input node
    fourth_node= Node(id='dense3_layer', metadata= {'color':'.8 .8 .8'})
    fourth_node.input_ports.append(InputPort(id= 'input'))
    fourth_node.parameters.append(Parameter(id='dense3_weight', value= params['dense']['weights']))
    fourth_node.parameters.append(Parameter(id='dense3_bias', value= params['dense']['bias']))
    feedForward= Parameter(id= 'feedForward',value= '(dense3_weight) + dense3_bias')
    fourth_node.parameters.append(feedForward)
    fourth_node.output_ports.append(OutputPort(id= 'dense3_output', value= 'feedForward trained network'))
    graph.nodes.append(fourth_node)
    simple_connect(activate_3, fourth_node, graph)

    # creating activation for the fourth node and,appending to the graph and connect it to input layer node
    activate_4= Node(id= 'dense_activation')
    activate_4.input_ports.append(InputPort(id='input'))
    activation= Parameter(id= 'relu',value = 'input*input')
    
    activation.conditions.append(ParameterCondition(id= 'test',
                            test= 'relu=0', value = 'relu'))

    activate_4.parameters.append(activation)
    activate_4.output_ports.append(OutputPort(id='activate_4', value= 'relu' ))
    graph.nodes.append(activate_4)
    simple_connect(fourth_node, activate_4, graph)
    
    # create fifth layer node, append to graph and connect it to second layer activation node

    fifth_node= Node(id='dense4_layer', metadata= {'color':'.8 .8 .8'})
    fifth_node.input_ports.append(InputPort(id= 'input'))
    fifth_node.parameters.append(Parameter(id='dense4_weight', value= params['dense_4']['weights']))
    fifth_node.parameters.append(Parameter(id='dense4_bias', value= params['dense_4']['bias']))
    feedForward= Parameter(id= 'feedForward',
                           value= '(dense4_weight) + dense4_bias')
    fifth_node.parameters.append(feedForward)

    fifth_node.output_ports.append(OutputPort(id= 'dense4_output', value= 'feedForward'))
    graph.nodes.append(fifth_node)
    simple_connect(activate_4, fifth_node, graph)
    
    # create fifth layer activation node, append to graph and connect it to fifth layer node

    activate_5 = Node(id= 'dense4_activation')
    activate_5.input_ports.append(InputPort(id='input'))
    activation= Parameter(id= 'sigmoid', 
                         value= '1/(1 + (2.71828**(-input)))' )
    activate_5.parameters.append(activation)
    activate_5.output_ports.append(OutputPort(id='activate_5', value= 'sigmoid' ))
    graph.nodes.append(activate_3)
    simple_connect(fifth_node, activate_5, graph)

    # return keras model that has been converted to mdf model and its equivalent graph
    
    return model_mdf, 