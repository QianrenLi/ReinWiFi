import test_case as tc
import os
import numpy as np
import matplotlib.pyplot as plt
import copy
import argparse
import traceback
import csv
import time
import threading
import torch

from training.environment import envCap
from training.system2trainV3 import wlanRLController
from training.imitator import imitator
from transmission_graph import Graph



abs_path = os.path.dirname(os.path.abspath(__file__))
DURATION = 10

FRACTION = "fraction"
CW = "cw"
AIFS = "aifs"

env_change_factor = 0.005


def add_file(graph: Graph, link_name:str, port:int):
    graph.ADD_STREAM(
        link_name,
        port_number=port,
        file_name="file_75MB.npy",
        duration=[0, DURATION],
        thru=0,
        tos=96,
        name="File",
    )

def add_proj(graph: Graph, link_name:str, port:int, target_rtt = 18):
    graph.ADD_STREAM(
        link_name,
        port_number=port,
        file_name="proj_6.25MB.npy",
        duration=[0, DURATION],
        thru = 3.5 * 8,
        tos= 128,
        target_rtt= target_rtt,
        name="Proj-%s" % link_name,
    )

def add_cast(graph: Graph, link_name:str, port:int, target_rtt = 24):
    graph.ADD_STREAM(
        link_name,
        port_number=port,
        file_name="proj_6.25MB.npy",
        duration=[0, DURATION],
        thru = 3.5 * 8 / 2,
        tos= 128,
        target_rtt= target_rtt,
        name="Proj-%s" % link_name,
    )


def compute_cost(graph: Graph, cost_threshold=100):
    """
    Compute cost for each iteration
    """
    cost = 10
    his_fraction = 0
    for device_name, links in graph.graph.items():
        for link_name, streams in links.items():
            for stream_name, stream in streams.items():
                target_rtt = graph.info_graph[device_name][link_name][
                    stream_name
                ]["target_rtt"]
                if (
                    graph.info_graph[device_name][link_name][stream_name][
                        "active"
                    ]
                    == True
                ):
                    # cost += 1
                    if "file" in stream["file_name"]:
                        his_fraction += graph.info_graph[device_name][link_name][stream_name]["fraction"]
                    if target_rtt != 0 and stream["rtt"] is not None:
                        # cost += stream["rtt"] * 1000 > target_rtt
                        _cost = max(stream["rtt"] * 1000 - target_rtt, 0)
                        cost += _cost if _cost < cost_threshold else cost_threshold
    cost -= his_fraction * 10
    # print("cost", cost)
    return cost


def action2ind(controls, action_space, graph:Graph):
    index = 0
    fraction = controls[FRACTION]

    his_num = len(action_space[0])
    cw_aifs_num = [len(action_space[1]),  len(action_space[2])]
    fraction_idx = np.where(np.array(action_space[0]) == fraction)[0][0]
    index += fraction_idx

    # cw_idx = np.where(np.array(action_space[1]) == cw)[0][0]
    # aifs_idx = np.where(np.array(action_space[2]) == aifs)[0][0]

    for device_name, links in graph.graph.items():
        for link_name, streams in links.items():
            for stream_name, stream in streams.items():
                prot, sender, receiver = link_name.split("_")
                if "file" not in stream["file_name"]:
                    port, tos = stream_name.split("@")
                    control_name = prot + "_" + tos + "_" + sender + "_" + receiver
                    for _idx, action_name in enumerate(["cw", "aifs"]):
                        _val = controls[control_name][action_name]
                        _ind = np.where(np.array(action_space[ _idx + 1 ]) == _val)[0][0]
                        index += _ind * his_num
                        his_num *= cw_aifs_num[_idx]
    index = max(index, 1)
    return index



def ind2state(env:envCap, ind):
    state = copy.deepcopy(env.env_graph[env.scenario]["system_return"][str(ind)])
    for stream_name in state:
        for port_name in state[stream_name]:
            if "rtt" in state[stream_name][port_name]:
                if state[stream_name][port_name]["rtt"] > 0:
                    noise_rtt = state[stream_name][port_name]["rtt"] + np.random.normal(0, 0.0005)
                    # print(noise_rtt)
                    state[stream_name][port_name].update({"rtt": noise_rtt})
    # TODO: add random noise
    return state

def store_trainsition(state,original_action, action_idx, cost, state_, wlanController:wlanRLController, rl_kernel:str):
    if rl_kernel == "DQN":
        wlanController.store_transition(state, action_idx, cost, state_)
    else:
        wlanController.store_transition(state, original_action, cost, state_)


def evaluation(wlanController:wlanRLController, env):
    initial_ind = 1
    episode = 200

    action_space = [
        [i / 20 for i in range(1, 20, 1)],
        [1, 3 , 7 , 15, 31, 63],  # CW value
        [1, 3, 5, 7, 9, 11]
    ]

    wlanController.init_action_guess()
    wlanController.graph.update_graph(ind2state(env, initial_ind))
    state = None
    state_ = None

    value = 0
    for _ in range(episode):
        if state is not None:
            state_ = wlanController.get_state()
        else:
            state = wlanController.get_state()
        if state_ is not None:
            cost = wlanController.get_cost(his_fraction)
            value += cost
            store_trainsition(state,original_action, action_idx, cost, state_, wlanController, args.rl)
        controls, action_idx,original_action = wlanController.action_to_control(state)
        his_fraction = controls[FRACTION]
        ## system transition
        index = action2ind(controls, action_space, wlanController.graph)
        wlanController.graph.update_graph(ind2state(env, index))
    print("Validation\t",  value / episode)
    return value / episode

def state_list2dict( graph, state_list , data_type = "s"):
    state_dict = {}
    idx = 0
    for device_name, links in graph.graph.items():
        for link_name, streams in links.items():
            prot, sender, receiver = link_name.split("_")
            state_dict.update({link_name:{}})
            for stream_name, stream in streams.items():
                port, tos = stream_name.split("@")
                state_dict[link_name].update({ stream_name : { } })
                if stream["thru"] != 0:
                    if data_type == "s":
                        state_dict[link_name][stream_name].update( { "rtt" : state_list[idx] / 1000 } )
                    else:
                        state_dict[link_name][stream_name].update( { "rtt" : state_list[idx] } )
                    state_dict[link_name][stream_name].update( {"throughput" : ((len(state_list) - idx - 1) // 2 + 1) * 26 } )
                    idx += 1
                else:
                    state_dict[link_name][stream_name].update( { "rtt" : 0 } )
                    state_dict[link_name][stream_name].update( {"throughput" : 200 } )
    assert( idx == len(state_list) )
    return state_dict

def evaluation_env(wlanController, env_imitator, initial_action, graph , display = False):
    state = None
    state_ = None
    value = 0
    initial_state =  env_imitator.predict( initial_action )[0]
    initial_state_dict = state_list2dict(graph, initial_state)
    # wlanController.graph.show()
    wlanController.init_action_guess()
    wlanController.graph.update_graph(initial_state_dict)
    episode = 200
    # wlanController.graph.show()
    costs = []
    for i in range(episode):
        if state is not None:
            state_ = wlanController.get_state()
        else:
            state = wlanController.get_state()
        if state_ is not None:
            cost = wlanController.get_cost(his_fraction)
            # print(cost)
            # exit()
            costs.append(cost)
            value += cost
            store_trainsition(state, original_action, action_idx, cost, state_, wlanController, args.rl)
            state = state_
        # print(state)
        controls, action_idx, original_action = wlanController.action_to_control(state)
        his_fraction = controls[FRACTION]
        original_action = [i for i in original_action if i != 0]
        system_return_env = env_imitator.predict( original_action )[0]
        if display == True:
            print(original_action)
        wlanController.graph.update_graph(state_list2dict( graph, system_return_env))
    print(sum( np.array(costs) < 10 )/ len(costs))
    print(state)
    print("Validation\t",  value / episode)
    return value / episode

def train_controller_csv( args ):
    # for interference_idx, env_path in enumerate(["/env-8-3-1.pt", "/env-8-3-2.pt", "/env-8-3-3.pt"]):
    np.random.seed(34)
    torch.manual_seed(34)
    graph, lists = tc.cw_training_case()
    env = envCap(graph, abs_path + "/training/env/env4.json")
    env.scenario = "scenario_4"
    add_proj(graph, lists[2], 6201)
    add_file(graph, lists[2], 6200)

    wlanController = wlanRLController(
        [i / 20 for i in range(1, 20, 1)],
        [1, 3 , 7 , 15, 31, 63],  # CW value
        [2],  # AIFSN
        100000,
        graph,
        batch_size = 64 ,
        is_CDQN = False,
        is_k_cost= 0,
        gamma = 0.9,
        log_level= 1
    )
    if args.test:
        try:
            wlanController.load_params("%s%s%s.pt" % (abs_path, args.ctlModelFolder, args.model))
        except:
            print("Parameter load fault")
        
    envPaths = args.model_paths
    ## start training
    num_episode = 1 if args.test else 200
    episode = 2000
    loss_curve = []
    loss_cum = 10
    wlanController.is_local_train = not args.test
    for _ in range(num_episode):
        np.random.shuffle(envPaths)
        for env_idx, env_path in enumerate(envPaths):
            print('------- ' + env_path)
            imitator_idx = int(env_path.split("/")[-1].split("_")[1][-1])
            graph, lists = tc.cw_training_case()
            wlanController.graph = graph
            if imitator_idx == 1:
                add_proj(graph, lists[2], 6201)
                add_file(graph, lists[2], 6200)
                action_num = 1
            elif imitator_idx == 2 or imitator_idx == 4:
                add_proj(graph, lists[2], 6201)
                add_file(graph, lists[2], 6200)
                add_cast(graph, lists[0], 6202)   
                action_num = 2
            elif imitator_idx == 3 or imitator_idx == 5:
                add_cast(graph, lists[1], 6204)
                add_proj(graph, lists[2], 6201)
                add_file(graph, lists[2], 6200)
                add_cast(graph, lists[0], 6202)
                action_num = 3

            env_imitator = imitator(env, graph, action_num * 2 + 1, action_num)
            env_imitator.load_params( abs_path + '/' +  env_path )
            env_imitator.data_type = "list"
            state = None
            state_ = None
            value = 0
            # initial_action = [0.1, 1,2, 1,2]
            initial_action = [0.1] + [1,2] * action_num
            initial_state =  env_imitator.predict( initial_action )[0]
            initial_state += np.random.random(len(initial_state)) - 0.5
            initial_state_dict = state_list2dict(graph, initial_state)
            wlanController.init_action_guess()
            wlanController.cluster_idx = env_idx
            wlanController.graph.update_graph(initial_state_dict)
            counter_max = 50
            loss = []
            
            if args.test or _ % 5 == 0:
                if args.test:
                    wlanController.action_counter = 10000
                evaluation_env(wlanController, env_imitator, initial_action = initial_action, graph = graph, display= False)
                continue
            # wlanController.graph.show()
            for i in range(int(episode  * np.exp(-_ / 2000))):
                if state is not None:
                    state_ = wlanController.get_state()
                else:
                    state = wlanController.get_state()
                if state_ is not None:
                    cost = wlanController.get_cost(his_fraction)
                    value += cost
                    store_trainsition(state, original_action, action_idx, cost, state_, wlanController, args.rl)
                    state = state_
                    
                controls, action_idx, original_action = wlanController.action_to_control(state)
                his_fraction = controls[FRACTION]
                original_action = [i for i in original_action if i != 0]
                system_return_env = env_imitator.predict( original_action )[0]
                wlanController.graph.update_graph(state_list2dict( graph, system_return_env))

                if wlanController.memory_counter > 100 and args.test == False:
                    [loss.append( wlanController.training_network() ) for _ in range(1)] ## 1 trail 10 training
                    if len(loss) > counter_max:
                        print("loss\t", sum(loss) / counter_max)
                        loss_curve += loss
                        loss.clear()
            # evaluation_env(wlanController, env_imitator, initial_action = initial_action, graph = graph)
            print("value\t", value / episode)
            wlanController.store_params("%s%s%s.pt" % (abs_path, args.ctlModelFolder , args.model))
            print("------- Model saved")

def train_env_csv( args ):
    env_idx = 0
    env_files = args.env_files
    model_paths = args.model_paths
    assert(len(env_files) == len(model_paths))
    COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for env_path, model_path in zip(env_files, model_paths):
        graph, lists = tc.cw_training_case()
        env = envCap(graph, abs_path + "/training/env/env4.json")
        env.scenario = "scenario_4"
        add_proj(graph, lists[0], 6201)
        add_proj(graph, lists[1], 6202)
        add_cast(graph, lists[2], 6203)
        add_cast(graph, lists[3], 6204)
        add_file(graph, lists[2], 6200)
        rtt_maximum = 40
        ## Data Read
        import csv
        env_idx += 1
        action = []
        rtt = []
        path = abs_path  + '/' + env_path
        with open( path, "rt" ) as f:
            cr = csv.reader(f)
            for idx, row in enumerate(cr):
                float_row = []
                for element in row:
                    val = float(element)
                    if idx % 2 == 0:
                        val = rtt_maximum if val > rtt_maximum else val
                    float_row.append(val)
                if idx % 2 == 0:
                    rtt.append(float_row[:-1])
                    print(float_row[:-1])
                else:
                    action.append(float_row)
        assert(len(action) == len(rtt))
        # print(rtt)
        env_imitator = imitator(env, graph, len(action[0]), len(rtt[0]))
        env_imitator.data_type = "list"

        ## Data set divide
        np.random.seed(32)
        torch.manual_seed(32)

        print( len(action[0]) )

        parts = 30
        indices = np.array(list(range(len(action))))
        np.random.shuffle(indices)
        indices_len = len(indices)
        train_indices = indices[: int((parts - 1)/ parts * indices_len)]
        validation_indices = indices[int((parts - 1)/ parts * indices_len):]

        ## Training
        epoch = 500 # if env_path != f'/env_s{senario_idx}_3.csv' else 1500
        # batch_num = 64
        batch_num = 64
        loss_cum = len(train_indices) // batch_num

        loss_curve = []
        validation_curve = []
        his_validation_loss = 10000
        for i in range(epoch):
            try:
                loss = 0
                counter = 0
                loss_counter = 0
                controls = []
                system_returns = []
                # Select from training indices
                for index in train_indices:
                    control = action[index]
                    system_return = rtt[index]

                    if counter < batch_num:
                        controls.append(control)
                        system_returns.append(system_return)
                        counter += 1
                    else:
                        counter = 0
                        _loss = env_imitator.train( controls, system_returns )
                        loss += _loss
                        loss_counter += 1
                        controls = []
                        system_returns = []
                    
                        loss_curve.append(_loss)
                
                ## validation
                loss = 0
                for index in validation_indices:
                    # print(index)
                    control = action[index]
                    system_return = rtt[index]
                    loss += env_imitator.eval( control, system_return )
                
                ## store
                if loss / len(validation_indices) < his_validation_loss:
                    his_validation_loss = loss / len(validation_indices)
                    env_imitator.store_params( abs_path + '/' + model_path )

                validation_curve.append(loss / len(validation_indices))

                print("%d-th epoch\t validation: \t %f" % (i, loss / len(validation_indices)))
            except KeyboardInterrupt:
                break
            
def interference_idx_classifier( args ):
    graph, lists = tc.cw_training_case()
    env = envCap(graph, abs_path + "/training/env/env4.json")
    env.scenario = "scenario_4"
    add_proj(graph, lists[0], 6201)
    add_proj(graph, lists[1], 6202)
    add_cast(graph, lists[2], 6203)
    add_cast(graph, lists[3], 6204)
    add_file(graph, lists[2], 6200)
    wlanController = wlanRLController(
        [i / 20 for i in range(1, 20, 1)],
        [1, 3 , 7 , 15, 31, 63],  # CW value
        [1, 3, 5, 7, 9, 11],  # AIFSN
        100000,
        graph,
        batch_size = 64 ,
        is_CDQN = False,
        is_k_cost= 0,
        gamma = 0.99
    )    
    action_rtt = []
    if_ides = []
    actions = []
    rtts = []
    ##
    env_paths = args.env_files
    for if_idx, env_path in enumerate(["/env/env_t1_1.csv", "/env/env_t1_2.csv", "/env/env_t1_3.csv"]):
    # for if_idx, env_path in enumerate(["/env/env_t2_1.csv", "/env/env_t2_2.csv", "/env/env_t2_3.csv"]):
        import csv
        action = []
        rtt = []
        path = abs_path + "/training" + env_path
        with open( path, "rt" ) as f:
            cr = csv.reader(f)
            for idx, row in enumerate(cr):
                float_row = []
                for element in row:
                    val = float(element)
                    float_row.append(val)
                if idx % 2 == 0:
                    rtt.append(float_row[:-1])
                else:
                    action.append(float_row)
        for i, j in zip(action, rtt):
            action_rtt.append(i + j)
            if if_idx == 0:
                if_ides.append([1,0,0])
            elif if_idx == 1:
                if_ides.append([0,1,0])
            else:
                if_ides.append([0,0,1])
        actions.append(action)
        rtts.append(rtt)
    env_imitator = imitator(env, graph, len(action_rtt[0]), 3)
    print(len(action_rtt[0]))
    env_imitator.data_type = "list"
    # env_imitator.load_params( abs_path + "/training/model/cfy_2.pt" )
    error_num = [0,0,0]
    if args.test:
        env_imitator.load_params( abs_path + "/training/model/cfy_2.pt" )
        correct = 0
        for a_idx, ar in enumerate(action_rtt):
            if_index = env_imitator.predict(ar)[0]
            if np.argmax(if_index) == np.argmax(if_ides[a_idx]):
                correct += 1
            else:
                error_num[np.argmax(if_ides[a_idx])] += 1
        print(error_num)
        print(correct / len(action_rtt))
        exit()

    ## Data set divide
    np.random.seed(1)
    torch.manual_seed(1)



    parts = 50
    indices = np.array(list(range(len(action_rtt))))
    np.random.shuffle(indices)
    indices_len = len(indices)
    train_indices = indices[: int((parts - 1)/ parts * indices_len)]

    validation_indices = indices[int((parts - 1)/ parts * indices_len):]

    ## Training
    epoch = 100
    batch_num = 64
    loss_cum = 10

    loss_curve = []
    validation_curve = []
    his_validation_loss = 50
    for i in range(epoch):
        try:
            loss = 0
            counter = 0
            loss_counter = 0
            controls = []
            system_returns = []
            # Select from training indices
            for index in train_indices:
                control = action_rtt[index]
                system_return = if_ides[index]
                
                if counter < batch_num:
                    controls.append(control)
                    system_returns.append(system_return)
                    counter += 1
                else:
                    # exit()
                    counter = 0
                    loss += env_imitator.train( controls, system_returns )
                    loss_counter += 1
                    controls = []
                    system_returns = []
            loss_curve.append(loss / loss_counter)
            
            ## validation
            loss = 0
            for index in validation_indices:
                # print(index)
                control = action_rtt[index]
                system_return = if_ides[index]
                loss += env_imitator.eval( control, system_return )
            
            ## store
            if loss / len(validation_indices) < his_validation_loss:
                his_validation_loss = loss / len(validation_indices)
                env_imitator.store_params( abs_path + "/training/model/cfy_1.pt" )

            validation_curve.append(loss / len(validation_indices))
            print("validation: \t %f" % (loss / len(validation_indices)))
            correct = 0

        except KeyboardInterrupt:
            break
        for a_idx, ar in enumerate(action_rtt):
            if_index = env_imitator.predict(ar)[0]
            if np.argmax(if_index) == np.argmax(if_ides[a_idx]):
                correct += 1

        print(correct / len(action_rtt))
    plt.plot(loss_curve, label = "Training Loss")
    plt.plot(validation_curve, label = "Validation Loss")
    plt.title("Training of Environment Imitator")
    plt.legend()
    plt.grid(True)
    plt.show()

def main(args):
    if args.trainEnv:
        train_env_csv( args )
    if args.trainController:
        train_controller_csv( args )

if __name__ == "__main__":
    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument("--config", is_config_file=True, default="./config/trainHis.txt")
    parser.add_argument("--scenario", type = int, nargs="+", help="scenario index in [1,2,3]")
    parser.add_argument("--env-files", type = str, nargs="+", help="env files", default=["/env_s1_1.csv", "/env_s1_2.csv", "/env_s1_3.csv"])
    parser.add_argument("--model-paths", type = str, nargs="+", help="model paths", default=["/env-s1-1.pt", "/env-s1-2.pt", "/env-s1-3.pt"])
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--model", type=str, default="/7-5")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--sort", action="store_true", help="sort the action and rtt in order when training env imitator")
    parser.add_argument("--rl", type=str, default="DQN")
    parser.add_argument("--envFolder", type=str, default="/training/env")
    parser.add_argument("--envModelFolder", type=str, default="/training/model", help="Folder store the training model")
    parser.add_argument("--ctlModelFolder", type=str, default="/training/model", help="Folder store the training model")
    parser.add_argument("--trainEnv", action="store_true", help="Train env imitator")
    parser.add_argument("--trainController", action="store_true", help="Train controller")
    parser.add_argument("--noInterference", action="store_true", help="if indicator")
    parser.add_argument("--cw-interference", action="store_true", help="Simulate the interference on CW value")
    args = parser.parse_args()
    main(args)
