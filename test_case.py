from transmission_graph import Graph

def name_to_thru(file_name):
    # extract throughput from file name
    file_size = float(file_name.split('_')[1].split('MB')[0])
    return file_size

def get_scenario_local_test(DURATION):
    graph = Graph()
    graph.ADD_DEVICE('phone')
    graph.ADD_DEVICE('PC')

    link1 = graph.ADD_LINK('phone', 'PC', 'lo', 1200)
    # link2 = graph.ADD_LINK('phone', 'PC', 'wlan', 866.7)
    link3 = graph.ADD_LINK('PC', 'phone', 'lo', 866.7)

    graph.ADD_STREAM(link1, port_number=list(range(5202, 5205)),
                     file_name="file_75MB.npy", duration=[0, DURATION], thru=0, name = "File")
    graph.ADD_STREAM(link1, port_number=6201, file_name="proj_6.25MB.npy", duration=[
                     0, DURATION], thru=name_to_thru("proj_6.25MB.npy"), tos=96, target_rtt= 0.3 , name = 'Proj')

    # graph.ADD_STREAM(link2, port_number=6202, file_name="voice_0.05MB.npy", duration=[
    #                  0, 10], thru=name_to_thru("voice_0.05MB.npy"))

    graph.ADD_STREAM(link3, port_number=6203, file_name="voice_0.05MB.npy", duration=[
                     0, DURATION], thru=name_to_thru("voice_0.05MB.npy"), tos=96, target_rtt= 0.1,name = 'Delay Sensitive')

    graph.associate_ip('PC', 'lo', '127.0.0.1')
    graph.associate_ip('phone', 'lo', '127.0.0.1')
    return graph


def get_scenario_test(DURATION):
    graph = Graph()
    graph.ADD_DEVICE('PC')
    graph.ADD_DEVICE('phone')

    link1 = graph.ADD_LINK('phone', 'PC', 'p2p', 1200)
    link2 = graph.ADD_LINK('phone', 'PC', 'wlan', 866.7)
    link3 = graph.ADD_LINK('PC', 'phone', 'wlan', 866.7)

    graph.ADD_STREAM(link1, port_number=list(range(5202, 5205)),
                     file_name="file_75MB.npy", duration=[0, 10], thru=0)


    return graph

def get_scenario_1_graph(DURATION):
    graph = Graph()
    graph.ADD_DEVICE('phone')
    graph.ADD_DEVICE('PC')

    link1 = graph.ADD_LINK('phone', 'PC', 'p2p', 1200)
    link2 = graph.ADD_LINK('phone', 'PC', 'wlan', 866.7)
    # link3 = graph.ADD_LINK('PC', 'phone', 'p2p', 866.7)
    link4 = graph.ADD_LINK('PC', 'phone', 'wlan', 866.7)

    graph.ADD_STREAM(link1, port_number=list(range(5202, 5205)),
                     file_name="file_75MB.npy", duration=[0, DURATION ], thru=0, name = "File")

    graph.ADD_STREAM(link1, port_number=6201, file_name="proj_6.25MB.npy", duration=[
                     0, DURATION ], thru=name_to_thru("proj_6.25MB.npy"), tos=96, target_rtt=18, name='Proj')

    graph.ADD_STREAM(link2, port_number=6202, file_name="voice_0.05MB.npy", duration=[
                     0, DURATION ], thru=name_to_thru("voice_0.05MB.npy"), tos=192, target_rtt=40,name='Speaker')


    graph.ADD_STREAM(link4, port_number=6204, file_name="voice_0.05MB.npy", duration=[
                     0, DURATION], thru=name_to_thru("voice_0.05MB.npy"), tos=192, target_rtt=40,name='Mic')
    return graph


def get_scenario_2_graph(DURATION):
    graph = Graph()
    graph.ADD_DEVICE('phone')
    graph.ADD_DEVICE('PC')
    graph.ADD_DEVICE('pad')

    link1 = graph.ADD_LINK('phone', 'PC', 'p2p', 1048)
    link2 = graph.ADD_LINK('phone', 'PC', 'wlan', 866.7)
    link3 = graph.ADD_LINK('PC', 'phone', 'wlan', 866.7)
    link4 = graph.ADD_LINK('PC', 'phone', 'p2p', 866.7)
    link5 = graph.ADD_LINK('PC', 'pad', 'p2p', 1200)

    graph.ADD_STREAM(link1, port_number=list(range(5201, 5206)),
                     file_name="file_75MB.npy", duration=[0, DURATION], thru=0, tos=32, name= 'File A')
    graph.ADD_STREAM(link2, port_number=6201, file_name="voice_0.05MB.npy", duration=[
                     0, DURATION], thru=name_to_thru("voice_0.05MB.npy"), tos=128, target_rtt=40,name= 'Speaker')

    graph.ADD_STREAM(link3, port_number=6202, file_name="voice_0.05MB.npy", duration=[
                     0, DURATION], thru=name_to_thru("voice_0.05MB.npy"), tos=128, target_rtt=40,name= 'Mic')

    graph.ADD_STREAM(link4, port_number=6203, file_name="kb_0.125MB.npy", duration=[
                     0, DURATION], thru=name_to_thru("kb_0.125MB.npy"), tos=128, target_rtt=40,name= 'Keyboard A')

    graph.ADD_STREAM(link5, port_number=list(range(5206, 5209)),
                     file_name="file_75MB.npy", duration=[0, DURATION], thru=0, tos=32,name= 'File B')
    graph.ADD_STREAM(link5, port_number=6204, file_name="kb_0.125MB.npy", duration=[
                     0, DURATION], thru=name_to_thru("kb_0.125MB.npy"), tos=128, target_rtt=40,name= 'Keyboard B')
    graph.ADD_STREAM(link5, port_number=6205, file_name="proj_6.25MB.npy", duration=[
                     0, DURATION], thru=name_to_thru("proj_6.25MB.npy"), tos=128, target_rtt=20,name= 'Proj')
    return graph

def get_scenario_3_graph(DURATION):
    graph = Graph()
    graph.ADD_DEVICE('phone')
    graph.ADD_DEVICE('PC')
    graph.ADD_DEVICE('pad')
    graph.ADD_DEVICE('TV')

    link1 = graph.ADD_LINK('phone', 'PC', 'p2p', 1048)
    link2 = graph.ADD_LINK('phone', 'PC', 'wlan', 866.7)
    link3 = graph.ADD_LINK('PC', 'phone', 'wlan', 866.7)
    link5 = graph.ADD_LINK('pad', 'TV', 'p2p', 1048)
    link6 = graph.ADD_LINK('TV', 'pad', 'p2p', 1048)

    graph.ADD_STREAM(link1, port_number=list(range(5201, 5204)),
                     file_name="file_75MB.npy", duration=[0, DURATION], thru=0, tos=32, name= 'File')
    graph.ADD_STREAM(link1, port_number=6201, file_name="proj_6.25MB.npy", duration=[
                     0, DURATION], thru=name_to_thru("proj_6.25MB.npy"), tos=128, target_rtt=18, name= 'Proj')

    # graph.ADD_STREAM(link1, port_number=6210, file_name="proj_6.25MB.npy", duration=[
    #                  30, 40], thru=name_to_thru("proj_6.25MB.npy"), tos=128, target_rtt=18, name= 'Proj')

    graph.ADD_STREAM(link2, port_number=6202, file_name="voice_0.05MB.npy", duration=[
                     0, DURATION], thru=name_to_thru("voice_0.05MB.npy"), tos=128, target_rtt=40, name= 'Speaker A' )

    graph.ADD_STREAM(link3, port_number=6203, file_name="voice_0.05MB.npy", duration=[
                     0, DURATION], thru=name_to_thru("voice_0.05MB.npy"), tos=128, target_rtt=40,  name= 'Mic A')

    graph.ADD_STREAM(link5, port_number=6204, file_name="voice_0.05MB.npy", duration=[
                     0, DURATION], thru=name_to_thru("voice_0.05MB.npy"), tos=128, target_rtt=25 ,name= 'Speaker B')
    graph.ADD_STREAM(link5, port_number=6205, file_name="proj_6.25MB.npy", duration=[
                     0, DURATION], thru=name_to_thru("proj_6.25MB.npy"), tos= 128, target_rtt=20, name= 'Display')

    graph.ADD_STREAM(link6, port_number=6206, file_name="voice_0.05MB.npy", duration=[
                     0, DURATION], thru=name_to_thru("voice_0.05MB.npy"), tos=128, target_rtt=25, name= 'Mic B')
    graph.ADD_STREAM(link6, port_number=6207, file_name="proj_6.25MB.npy", duration=[
                     0, DURATION], thru=name_to_thru("proj_6.25MB.npy"), tos= 128, target_rtt=20, name= 'Camera')

    
    return graph

def scenario3(DURATION):
    graph = Graph()
    graph.ADD_DEVICE('phone')
    graph.ADD_DEVICE('PC')
    graph.ADD_DEVICE('pad')
    graph.ADD_DEVICE('TV')

    link1 = graph.ADD_LINK('phone', 'PC', 'p2p', 1048)
    link2 = graph.ADD_LINK('phone', 'PC', 'wlan', 866.7)
    link3 = graph.ADD_LINK('PC', 'phone', 'wlan', 866.7)
    link5 = graph.ADD_LINK('pad', 'TV', 'p2p', 1048)
    link6 = graph.ADD_LINK('TV', 'pad', 'p2p', 1048)

    graph.ADD_STREAM(link1, port_number=list(range(5201, 5204)),
                     file_name="file_75MB.npy", duration=[0, DURATION], thru=0, tos=32, name= 'File')
    graph.ADD_STREAM(link1, port_number=6201, file_name="proj_6.25MB.npy", duration=[
                     0, DURATION], thru=name_to_thru("proj_6.25MB.npy"), tos=128, target_rtt=18, name= 'Proj')
    
    # graph.ADD_STREAM(link1, port_number=6210, file_name="proj_6.25MB.npy", duration=[
    #                  30, 40], thru=name_to_thru("proj_6.25MB.npy"), tos=128, target_rtt=18, name= 'Proj')

    graph.ADD_STREAM(link2, port_number=6202, file_name="voice_0.05MB.npy", duration=[
                     0, DURATION], thru=name_to_thru("voice_0.05MB.npy"), tos=128, target_rtt=40, name= 'Speaker A' )

    graph.ADD_STREAM(link3, port_number=6203, file_name="voice_0.05MB.npy", duration=[
                     0, DURATION], thru=name_to_thru("voice_0.05MB.npy"), tos=128, target_rtt=40,  name= 'Mic A')

    graph.ADD_STREAM(link5, port_number=6204, file_name="voice_0.05MB.npy", duration=[
                     0, DURATION], thru=name_to_thru("voice_0.05MB.npy"), tos=128, target_rtt=25 ,name= 'Speaker B')
    graph.ADD_STREAM(link5, port_number=6205, file_name="proj_6.25MB.npy", duration=[
                     0, DURATION], thru=name_to_thru("proj_6.25MB.npy"), tos=128, target_rtt=20, name= 'Display')

    graph.ADD_STREAM(link6, port_number=6206, file_name="voice_0.05MB.npy", duration=[
                     0, DURATION], thru=name_to_thru("voice_0.05MB.npy"), tos=128, target_rtt=25, name= 'Mic B')
    graph.ADD_STREAM(link6, port_number=6207, file_name="proj_6.25MB.npy", duration=[
                     0, DURATION], thru=name_to_thru("proj_6.25MB.npy"), tos=128, target_rtt=20, name= 'Camera')    
    return graph

def scenario3_add_file(graph: Graph,DUARTION: int) -> Graph:
    graph.ADD_STREAM('p2p_phone_PC', port_number=6208, file_name="file_75MB.npy", duration=[20,DUARTION], thru=0, tos=32, name= 'Add-File')
    return graph

def scenario3_add_proj(graph: Graph, DUARTION: int):
    graph.ADD_STREAM('p2p_phone_PC', port_number=6208, file_name="proj_6.25MB.npy", duration=[
                     20, DUARTION], thru=name_to_thru("proj_6.25MB.npy"), tos=128, target_rtt= 30, name= 'Add-Proj')    
    return graph

def scenario3_add_interference(graph: Graph):
    return graph

def scenario3_remove_proj(graph: Graph):
    graph.REMOVE_STREAM('p2p_phone_PC', port_number=6201, tos=128)
    graph.ADD_STREAM('p2p_phone_PC', port_number=6208, file_name="proj_6.25MB.npy", duration=[
                     0, 30], thru=name_to_thru("proj_6.25MB.npy"), tos=128, target_rtt=18, name= 'Remove-Proj')
    return graph

def scenario3_remove_interference(graph: Graph) -> Graph: 
    return graph

def cw_test_case(DURATION) -> Graph:
    graph = Graph()
    graph.ADD_DEVICE('phone')
    graph.ADD_DEVICE('PC')    
    link1 = graph.ADD_LINK('phone', '', 'wlx', 800)
    link2 = graph.ADD_LINK('PC', '', 'wlp', 600)

    # graph.ADD_STREAM(link1, port_number=6202, file_name="proj_6.25MB.npy", duration=[
    #                  0, DURATION], thru=name_to_thru("proj_6.25MB.npy"), tos=128, target_rtt=18, name= 'Proj')

    # graph.ADD_STREAM(link2, port_number=6203, file_name="proj_6.25MB.npy", duration=[
    #                  0, DURATION], thru=name_to_thru("proj_6.25MB.npy"), tos=128, target_rtt=18, name= 'Proj')

    graph.ADD_STREAM(link1, port_number=6202, file_name="file_75MB.npy", duration=[
                     0, DURATION], thru=0, tos=32, name= 'File')   
    graph.ADD_STREAM(link2, port_number=6201, file_name="file_75MB.npy", duration=[
                     0, DURATION], thru=0, tos=32, name= 'File')   
    return graph

def cw_training_case(*args):
    if len(args) == 1:
        MCSs = args[0]
    else:
        MCSs = [600, 600]
    graph = Graph()
    graph.ADD_DEVICE('STA-1')
    graph.ADD_DEVICE('STA-2')
    graph.ADD_DEVICE('STA-3')
    graph.ADD_DEVICE('STA-4')
    graph.ADD_DEVICE('SoftAP')
    # graph.ADD_DEVICE('TV')

    link1 = graph.ADD_LINK('STA-1', 'SoftAP', 'eth', MCSs[0])
    link2 = graph.ADD_LINK('STA-2', 'SoftAP', 'eth', MCSs[1]) 
    link3 = graph.ADD_LINK('STA-3', 'SoftAP', 'eth', MCSs[0])
    link4 = graph.ADD_LINK('STA-4', 'SoftAP', 'eth', MCSs[1]) 
    
    # # link4 = graph.ADD_LINK('TV', '', 'wlx', 400)    
    # lists = [link1,link2]
    # lists = [link1,link2]
    lists = [link1,link2, link3,link4]
    # lists = [link1, link2, link4]
    return graph, lists

def cw_training_case_without_intereference(*args):
    if len(args) == 1:
        MCSs = args[0]
    else:
        MCSs = [600, 600]
    graph = Graph()
    graph.ADD_DEVICE('STA-1')
    graph.ADD_DEVICE('STA-2')
    graph.ADD_DEVICE('STA-3')
    graph.ADD_DEVICE('SoftAP')
    # graph.ADD_DEVICE('TV')

    link1 = graph.ADD_LINK('STA-1', 'SoftAP', 'eth', MCSs[0])
    link2 = graph.ADD_LINK('STA-2', 'SoftAP', 'eth', MCSs[1]) 
    link3 = graph.ADD_LINK('STA-3', 'SoftAP', 'eth', MCSs[0])
    
    # # link4 = graph.ADD_LINK('TV', '', 'wlx', 400)    
    # lists = [link1,link2]
    # lists = [link1,link2]
    lists = [link1,link2, link3]
    # lists = [link1, link2, link4]
    return graph, lists
            



if __name__ == '__main__':
    cw_training_case(2)