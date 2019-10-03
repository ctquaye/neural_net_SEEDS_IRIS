import math
import sys
import os
import numpy as np

Matrix =[]
input= []
output=[]
Container = []
bias =[]
sigmoid_arr =[]
# Sigmoid function definition
def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def Forward_Propagation(inputs_to_layer,counter,neuron,weighted_sum,temp_input,remover):
    while (counter < len(Matrix)):
        inputs_to_layer = 0
        weighted_sum = 0
        while (inputs_to_layer < len(input)):
            weighted_sum = weighted_sum + (input[inputs_to_layer] * Matrix[neuron][inputs_to_layer + 2])
            inputs_to_layer = inputs_to_layer + 1
        to_sigmoid = weighted_sum + bias[neuron]


        # for first neuron in first layer
        if neuron == 0:
            for i in range(len(input)-1):
                temp_input.append(input[i])
                sigmoid_arr.append(sigmoid(input[i]))
            temp_input.append(sigmoid(to_sigmoid))
            sigmoid_arr.append(sigmoid(to_sigmoid))
        # for subsequent neurons
        elif neuron > 0:
            temp_input.append(sigmoid(to_sigmoid))
            sigmoid_arr.append(sigmoid(to_sigmoid))
        else:
            pass
        if counter < len(Matrix) and neuron != len(Matrix) - 1:

            # if within the same layer
            if Matrix[neuron][0] == Matrix[neuron + 1][0]:
                neuron = neuron + 1


            # if moving to next layer
            elif Matrix[neuron][0] <= Matrix[neuron + 1][0]:
                remover = inputs_to_layer
                while (remover != 0 and remover != len(temp_input)):
                    temp_input.remove(temp_input[0])
                    remover = remover - 1
                input.clear()
                for i in range(len(temp_input)):
                    input.append(temp_input[i])
                neuron = neuron + 1
            counter = counter + 1

        else:
            counter = counter + 1

    remover = inputs_to_layer
    while (remover != 0 and remover != len(temp_input)):
        temp_input.remove(temp_input[0])
        remover = remover - 1


    return temp_input

count = 0
#read stdin from terminal
neural_net_descr_file = sys.argv[1]
input_filename = sys.argv[2]
low_train_rng = sys.argv[3]
hi_train_rng = sys.argv[4]
low_test_rng = sys.argv[5]
hi_test_rng = sys.argv[6]
num_epochs = sys.argv[7]
print_internals_flag = sys.argv[8]

with open(input_filename,"r") as infile:
    #header = infile.readline()
    data = []
    for line in infile:
        sline = line.strip().split(',')
        sline[-1] = int(sline[-1])
        for idx in range(len(sline)-1):
            sline[idx] = float(sline[idx])
        data.append(sline[0:len(sline)])
    labels = sorted( set( [sline[-1] for sline in data] ) )
    # create onehots for labels

    onehots = []
    for sline in data:
        onehot = [0] * len(labels)
        #very smart       onehot [0,0,0] onehot[label->[0,1,2].label[whatever is there]] =1
        onehot[labels.index(sline[-1])] = 1
        onehots.append(onehot)
    labels = onehots
    # normalize data

    numInputs = len(data[0]) - 1  # all but labels
    for colidx in range( numInputs ):
        tot = 0.0
        maxval = 0.0
        minval = 999999.0
        for row in data:
            colval = row[colidx]
            if colval < minval:
                minval = colval
            if colval > maxval:
                maxval = colval
        diffval = maxval - minval
        for row in data:
            row[colidx] = (row[colidx] - minval) / diffval


#pick file name from stdin again to read network file------------------------
with  open(neural_net_descr_file, "r") as network_file :
    for line in network_file:
        # print("starting line:", count)
        for character in line.split():
            # skipping to next line right before comments start or skipping lines starting with comments
            if (character == "#" or line.startswith("#")):
                break

            else:
                where_to_cut = line.find('#')
                line = line[0:where_to_cut]
                Container = line.split()
                Matrix.append(Container)
                x = float(Container[len(Container)-1])
                bias.append(x)
                break

            #print(character)
        count = count + 1
   # print('This is my input\n', input)

    for row in range(len(Matrix)):
       for column in range (len(Matrix[row])):
            string_a = Matrix[row][column]
            Matrix[row][column]= float(string_a)
       #     print(Matrix[row][column], end=' ')
      # print()


for epoch in range (int(num_epochs)):
    if ((int(epoch)) == 0 or (int(epoch)) >= int(num_epochs)-1):
        if (int(print_internals_flag) == 1):
            ("INTERNAL FLAGS BEFORE TRAINING:\n")
            # INTERNAL FLAGS
            for row in range(len(Matrix)):
                for column in range(len(Matrix[row])):
                    #string_a = Matrix[row][column]
                    #Matrix[row][column] = float(string_a)
                    print(Matrix[row][column], end=' ')
                print()
    correctTrain = 0
    for train in range(int(hi_train_rng) - int(low_train_rng)):
        train_rng=int(hi_train_rng) - int(low_train_rng)
        inputs_to_layer = 0;  counter = 0;    neuron = 0;    weighted_sum = 0
        temp_input = [];    remover = 0

        #input=data[train]

        for iterate in range (len(data[train])):
            input.append(data[train][iterate])


        #FowardPropagate----------------------------------------------------------
        output = Forward_Propagation\
            (inputs_to_layer,counter,neuron,weighted_sum,temp_input,remover)
        #-----------------------------

        # BackPropagate----------------------------------------------------------
            #pick layer number for output layer
        isOutputLayer = Matrix[len(Matrix)-1][0]
        current_layer = Matrix[len(Matrix) - 1][0]
        current_neuron = Matrix[len(Matrix)-1][1]
        bp_neuron = Matrix[len(Matrix)-1][1]
        sig = -(len(temp_input) + 1)
        level = len(Matrix) - 1
        Hidden_delta_arr = []
        Output_delta_arr = []
        to_sig=0;   to_sig_hid=0;   delta=0;    w = -2;    d = -1;
        to_hid_delta=0; to_hid_delta_arr1=[]; to_hid_delta_arr2=[]; offset=2

        differences = []
        differences = np.subtract(labels[train],output)

        buffer = 0
        max_out = max(output)
        for d in range(len(output)):
            if(np.array_equal(output[output.index(max_out)],output[d])):
                buffer = buffer + 1
        if(buffer == len(output)):
            for kk in range (len(output)):
                if (kk == 0):
                    output[kk]=1
                else:
                    output[kk]=0
        elif(buffer == len(output)-1 and output[1]==output[2]):
            for dd in range (len(output)):
                if (dd == 1):
                    output[dd]=1
                else:
                    output[dd]=0
        else:
            output[output.index(max_out)] = 1
            for x in range(len(output)):
                if output[x] != 1:
                    output[x] = 0


        if (np.array_equal(output, onehots[train]) is True):
            correctTrain = correctTrain + 1

        input_len = len(data[0]) - 1
        output_out_len = len(temp_input)
        hidden_out_len = len(sigmoid_arr) - (input_len + output_out_len)
        input_chunk = data[train][0:input_len]
        hidden_chunk = sigmoid_arr[input_len:input_len + hidden_out_len]
        output_chunk = sigmoid_arr[input_len + hidden_out_len:]

        # Backpropagation Code-----------------------------------------------------------
        #backpropagation code for output layer
        for ct in range(output_out_len):  # var is still equal to layer number of output layer
            for it in range(len(Matrix[len(Matrix) - 1][2:-1])):  # range of weights
                to_sig = to_sig + (hidden_chunk[it] * Matrix[ct][it])

            delta = sigmoid(to_sig)*(1-sigmoid(to_sig))*differences[ct]
            Output_delta_arr.append(delta)


        start_of_outlayer=hidden_out_len
        #print("start of out:", start_of_outlayer,"=", hidden_out_len)
        #backpropagation code for hidden layer
        for zl in range (start_of_outlayer) :
            for xl in range(len(Matrix[0][2:-1])):
                to_sig_hid = to_sig_hid + (data[train][xl]*Matrix[zl][xl+offset])
            to_hid_delta_arr1.append(sigmoid(to_sig_hid)*(1-sigmoid(to_sig_hid)))

        for fl in range(len(Matrix[start_of_outlayer][2:-1])):
            for tl in range (len(Matrix[start_of_outlayer:])):
               to_hid_delta = to_hid_delta +(np.multiply(Output_delta_arr[tl],Matrix[fl+offset][tl]))
            to_hid_delta_arr2.append(to_hid_delta)

        Hidden_delta_arr = np.multiply(to_hid_delta_arr1,to_hid_delta_arr2)

        #Update Weights Outer layer Weights
        start_of_outlayer = hidden_out_len

        # Update Weights Output layer Weights
        for rl in range (output_out_len) :
            for cl in range(len(Matrix[len(Matrix) - 1][2:-1])):  # range of weights
                Matrix[start_of_outlayer+rl][cl+offset]  = Matrix[start_of_outlayer+rl][cl+offset] + (hidden_chunk[cl] * Output_delta_arr[rl])

        for rf in range (hidden_out_len) :
            for cf in range(len(Matrix[0][2:-1])):  # range of weights
                Matrix[rf][cf+offset]  = Matrix[rf][cf+offset] + (input_chunk[cf] * Hidden_delta_arr[rf])


    #clear sigmoid array clear input array
        sigmoid_arr.clear(); temp_input.clear(); input.clear();

    if (((int(epoch)) % 10 == 0 and (int(epoch)) >= 10) or int(epoch)==0):
        print(epoch, ": ", correctTrain, "/", train_rng)


correctTest = 0
for test in range(int(hi_test_rng) - int(low_test_rng)):
        test_rng =int(hi_test_rng) - int(low_test_rng)
        inputs_to_layer = 0;
        counter = 0;
        neuron = 0;
        weighted_sum = 0
        temp_input = [];
        remover = 0

        # input=data[train]

        for iterater in range(len(data[test])):
            input.append(data[test][iterater])

        # FowardPropagate----------------------------------------------------------
        output = Forward_Propagation \
            (inputs_to_layer, counter, neuron, weighted_sum, temp_input, remover)

        #print("output, onehot", output, onehots[test], "test", test)
        buffer = 0
        max_out = max(output)
        for d in range(len(output)):
            if (np.array_equal(output[output.index(max_out)], output[d])):
                buffer = buffer + 1
        if (buffer == len(output)):
            for kk in range(len(output)):
                if (kk == 0):
                    output[kk] = 1
                else:
                    output[kk] = 0
        elif (buffer == len(output) - 1 and output[1] == output[2]):
            for dd in range(len(output)):
                if (dd == 1):
                    output[dd] = 1
                else:
                    output[dd] = 0
        else:
            output[output.index(max_out)] = 1
            for x in range(len(output)):
                if output[x] != 1:
                    output[x] = 0

        if(np.array_equal(output,onehots[test]) is True):
            correctTest = correctTest + 1

        sigmoid_arr.clear();
        temp_input.clear();
        input.clear();

print("\n TEST:",correctTest,"/",int(hi_test_rng) - int(low_test_rng))