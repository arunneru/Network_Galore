from nrnTemplate.CellTypes.CellTemplateCommon import MNcell, dINr
import matplotlib.pyplot as plt
import numpy as np
import random
import numpy.random as rnd
from neuron import h
import param_brick
import csv
import pickle
h.load_file('stdrun.hoc')


import time, datetime, os

par = param_brick.create_params()
RecAll = 2
dt = 0.01
time_end = 1500
std_on = 1
delay = 0
weight = 0.12e-1
threshold = 0


dinr2dinr_pC = 0.3
dinr2ain_pC = 0.3
dinr2cin_pC = 0.7

ain2dinr_pC = 0.2
ain2cin_pC = 0.2
ain2ain_pC = 0.2

cin2cin_pC = 0.2
cin2dinr_pC = 0.2
cin2ain_pC = 0.2

lower2upper_pC = 0.4

prb_ext_drive = 1.0 


celltypes = par.celltypes
# num_types = par.num_types
# halves = par.halves

vect_index = par.vect_index
left_index = par.left_index
right_index = par.right_index

vect_index_lower = par.vect_index_lower
left_index_lower = par.left_index_lower
right_index_lower = par.right_index_lower

vect_index_upper = par.vect_index_upper
left_index_upper = par.left_index_upper
right_index_upper = par.right_index_upper



w_config = par.w_config

cin_wfac = [1024000]
# dep_gaba_tau = np.arange(3.0,30.0,3.0)
dep_gaba_tau = [10]


def CellCreation(RecAll=0, varDt=False, sim_num=1, cin_wf=1500, dep_gaba_t=10):
    cell_list = []
    for i in range(len(celltypes)):
        for j in vect_index[i]:
            if celltypes[i] == "dinr":
                cell_list.append(dINr(RecAll=RecAll, theta=threshold)) 
            else:       
                cell_list.append(MNcell(RecAll=RecAll, theta=threshold))
            cell_list[-1].whatami = celltypes[i]
            cell_list[-1].type_num = i
            if j in left_index[i]:
                cell_list[-1].body_side = 1
            else:
                cell_list[-1].body_side = 2
            if celltypes[i] == "cin":
                cell_list[-1].parameters["dep_gaba_t"] = dep_gaba_t 
                cell_list[-1].set_params("cin_wf", cin_wf)
    return cell_list


## Synaptic connection between neural types
def connection(pre, post):
    key = pre.whatami + " -> " + post.whatami
    if key in w_config:
        specs = w_config[key]
        for spec in specs:
            if spec != None:
                syn_type = spec[0]
                w_mean = spec[1]
                w_std = spec[2]
                loc = spec[4]
                sect = spec[5]

                if w_std != 0.0:
                    weight = rnd.normal(w_mean, w_std*std_on)
                else:
                    weight = w_mean
                pre.connect(post, syn_type, syn_w=weight, delay=delay, loc=loc, sect=sect)
    else:
        print("Something wrong with this connection type\n")
        print(key)
        raise Exception("Connection created is not included in the list of possible connections")

## Create gap junctions between dINs
def GapJunction(cell_list):
    ind_dinr = celltypes.index("dinr")
    for i in left_index[ind_dinr]:
        for j in left_index[ind_dinr]:
            if i!=j:
                cell_list[i].connect(cell_list[j], "gap", gmax=0.2e-3)

    for i in right_index[ind_dinr]:
        for j in right_index[ind_dinr]:
            if i!=j:
                cell_list[i].connect(cell_list[j], "gap", gmax=0.2e-3)

def ext_syn_drive(cell_list, rate_spike_drive=100, sim_dur=5000, prb_ext_drive=prb_ext_drive, weight=0.012e3, delay=0.05):
    ind_dinr = celltypes.index("dinr")
    trgt_dinr = random.sample(left_index[ind_dinr], int(prb_ext_drive*len(left_index[ind_dinr])))
    for i in trgt_dinr:
        cell_list[i].connect_ext_drive(rate_spike_drive, sim_dur, "ampa", delay=delay, weight=weight, loc=0.5, sect="Adend3")

        cell_list[i].connect_ext_drive(rate_spike_drive, sim_dur, "nmda", delay=delay, weight=weight, loc=0.5, sect="Adend3")
        
        # cell_list[i].set_clamps()


    trgt_dinr = random.sample(right_index[ind_dinr], int(prb_ext_drive*len(right_index[ind_dinr])))
    for i in trgt_dinr:
        cell_list[i].connect_ext_drive(rate_spike_drive, sim_dur, "ampa", delay=delay, weight=weight, loc=0.5, sect="Adend3")

        cell_list[i].connect_ext_drive(rate_spike_drive, sim_dur,  "nmda", delay=delay, weight=weight, loc=0.5, sect="Adend3")
        
        # cell_list[i].set_clamps()

def CreateConfigAdj(cell_list):
    
    indices_dinr = [i for i, x in enumerate(celltypes) if x == "dinr"]
    indices_cin = [i for i, x in enumerate(celltypes) if x == "cin"]
    indices_ain = [i for i, x in enumerate(celltypes) if x == "ain"]
    
    print("indices_dinr", indices_dinr)
    all_cin_lower = left_index[indices_cin[0]]
    all_ain_lower = left_index[indices_ain[0]]
    all_dinr_lower = left_index[indices_dinr[0]]

    print("all_dinr_lower_left", left_index[indices_dinr[0]])
    print("all_dinr_lower_right", right_index[indices_dinr[0]])
    print("all_dinr_upper_left", left_index[indices_dinr[1]])
    print("all_dinr_upper_right", right_index[indices_dinr[1]])
          
    
    pre_dinr = all_dinr_lower
    for dnr in pre_dinr:

        post_cin = random.sample(all_cin_lower, int(dinr2cin_pC*len(all_cin_lower)))
        for cin in post_cin:
            connection(cell_list[dnr], cell_list[cin])
    
        post_ain = random.sample(all_ain_lower, int(dinr2ain_pC*len(all_ain_lower)))
        for ain in post_ain:
            connection(cell_list[dnr], cell_list[ain])

        post_dinr = random.sample(all_dinr_lower, int(dinr2dinr_pC*len(all_dinr_lower)))
        for target_dinr in post_dinr:
            connection(cell_list[dnr], cell_list[target_dinr])

    
    pre_ain = all_ain_lower
    for ain in pre_ain:

        post_cin = random.sample(all_cin_lower, int(ain2cin_pC*len(all_cin_lower)))
        for cin in post_cin:
            connection(cell_list[ain], cell_list[cin])
    
        post_ain = random.sample(all_ain_lower, int(ain2ain_pC*len(all_ain_lower)))
        for target_ain in post_ain:
            connection(cell_list[ain], cell_list[target_ain])

        post_dinr = random.sample(all_dinr_lower, int(dinr2dinr_pC*len(all_dinr_lower)))
        for dinr in post_dinr:
            connection(cell_list[ain], cell_list[dinr])

    pre_cin = right_index[indices_cin[0]]
    for cin in pre_cin:
    
        post_cin = random.sample(all_cin_lower, int(cin2cin_pC*len(all_cin_lower)))
        for target_cin in post_cin:
            connection(cell_list[cin], cell_list[target_cin])
    
        post_ain = random.sample(all_ain_lower, int(dinr2ain_pC*len(all_ain_lower)))
        for ain in post_ain:
            connection(cell_list[cin], cell_list[ain])

        post_dinr = random.sample(all_dinr_lower, int(dinr2dinr_pC*len(all_dinr_lower)))
        for dinr in post_dinr:
            connection(cell_list[cin], cell_list[dinr])
            
    
    all_cin_lower = right_index[indices_cin[0]]
    all_ain_lower = right_index[indices_ain[0]]
    all_dinr_lower = right_index[indices_dinr[0]]
    
    pre_dinr = all_dinr_lower
    for dnr in pre_dinr:

        post_cin = random.sample(all_cin_lower, int(dinr2cin_pC*len(all_cin_lower)))
        for cin in post_cin:
            connection(cell_list[dnr], cell_list[cin])
    
        post_ain = random.sample(all_ain_lower, int(dinr2ain_pC*len(all_ain_lower)))
        for ain in post_ain:
            connection(cell_list[dnr], cell_list[ain])

        post_dinr = random.sample(all_dinr_lower, int(dinr2dinr_pC*len(all_dinr_lower)))
        for target_dinr in post_dinr:
            connection(cell_list[dnr], cell_list[target_dinr])

    
    pre_ain = all_ain_lower
    for ain in pre_ain:

        post_cin = random.sample(all_cin_lower, int(ain2cin_pC*len(all_cin_lower)))
        for cin in post_cin:
            connection(cell_list[ain], cell_list[cin])
    
        post_ain = random.sample(all_ain_lower, int(ain2ain_pC*len(all_ain_lower)))
        for target_ain in post_ain:
            connection(cell_list[ain], cell_list[target_ain])

        post_dinr = random.sample(all_dinr_lower, int(dinr2dinr_pC*len(all_dinr_lower)))
        for dinr in post_dinr:
            connection(cell_list[ain], cell_list[dinr])

    pre_cin = left_index[indices_cin[0]]
    for cin in pre_cin:
    
        post_cin = random.sample(all_cin_lower, int(cin2cin_pC*len(all_cin_lower)))
        for target_cin in post_cin:
            connection(cell_list[cin], cell_list[target_cin])
    
        post_ain = random.sample(all_ain_lower, int(dinr2ain_pC*len(all_ain_lower)))
        for ain in post_ain:
            connection(cell_list[cin], cell_list[ain])

        post_dinr = random.sample(all_dinr_lower, int(dinr2dinr_pC*len(all_dinr_lower)))
        for dinr in post_dinr:
            connection(cell_list[cin], cell_list[dinr])

    
     
    all_cin_upper = left_index[indices_cin[1]]
    all_ain_upper = left_index[indices_ain[1]]
    all_dinr_upper = left_index[indices_dinr[1]]
    
    pre_dinr = all_dinr_upper
    for dnr in pre_dinr:

        post_cin = random.sample(all_cin_upper, int(dinr2cin_pC*len(all_cin_upper)))
        for cin in post_cin:
            connection(cell_list[dnr], cell_list[cin])
    
        post_ain = random.sample(all_ain_upper, int(dinr2ain_pC*len(all_ain_upper)))
        for ain in post_ain:
            connection(cell_list[dnr], cell_list[ain])

        post_dinr = random.sample(all_dinr_upper, int(dinr2dinr_pC*len(all_dinr_upper)))
        for target_dinr in post_dinr:
            connection(cell_list[dnr], cell_list[target_dinr])

    
    pre_ain = all_ain_upper
    for ain in pre_ain:

        post_cin = random.sample(all_cin_upper, int(ain2cin_pC*len(all_cin_upper)))
        for cin in post_cin:
            connection(cell_list[ain], cell_list[cin])
    
        post_ain = random.sample(all_ain_upper, int(ain2ain_pC*len(all_ain_upper)))
        for target_ain in post_ain:
            connection(cell_list[ain], cell_list[target_ain])

        post_dinr = random.sample(all_dinr_upper, int(dinr2dinr_pC*len(all_dinr_upper)))
        for dinr in post_dinr:
            connection(cell_list[ain], cell_list[dinr])

    pre_cin = right_index[indices_cin[1]]
    for cin in pre_cin:
    
        post_cin = random.sample(all_cin_upper, int(cin2cin_pC*len(all_cin_upper)))
        for target_cin in post_cin:
            connection(cell_list[cin], cell_list[target_cin])
    
        post_ain = random.sample(all_ain_upper, int(dinr2ain_pC*len(all_ain_upper)))
        for ain in post_ain:
            connection(cell_list[cin], cell_list[ain])

        post_dinr = random.sample(all_dinr_upper, int(dinr2dinr_pC*len(all_dinr_upper)))
        for dinr in post_dinr:
            connection(cell_list[cin], cell_list[dinr])
            

    all_cin_upper = right_index[indices_cin[1]]
    all_ain_upper = right_index[indices_ain[1]]
    all_dinr_upper = right_index[indices_dinr[1]]
    
    pre_dinr = all_dinr_upper
    for dnr in pre_dinr:
        post_cin = random.sample(all_cin_upper, int(dinr2cin_pC*len(all_cin_upper)))
        for cin in post_cin:
            connection(cell_list[dnr], cell_list[cin])
    
        post_ain = random.sample(all_ain_upper, int(dinr2ain_pC*len(all_ain_upper)))
        for ain in post_ain:
            connection(cell_list[dnr], cell_list[ain])

        post_dinr = random.sample(all_dinr_upper, int(dinr2dinr_pC*len(all_dinr_upper)))
        for target_dinr in post_dinr:
            connection(cell_list[dnr], cell_list[target_dinr])

    
    pre_ain = all_ain_upper
    for ain in pre_ain:

        post_cin = random.sample(all_cin_upper, int(ain2cin_pC*len(all_cin_upper)))
        for cin in post_cin:
            connection(cell_list[ain], cell_list[cin])
    
        post_ain = random.sample(all_ain_upper, int(ain2ain_pC*len(all_ain_upper)))
        for target_ain in post_ain:
            connection(cell_list[ain], cell_list[target_ain])

        post_dinr = random.sample(all_dinr_upper, int(dinr2dinr_pC*len(all_dinr_upper)))
        for dinr in post_dinr:
            connection(cell_list[ain], cell_list[dinr])

    pre_cin = left_index[indices_cin[1]]
    for cin in pre_cin:
    
        post_cin = random.sample(all_cin_upper, int(cin2cin_pC*len(all_cin_upper)))
        for target_cin in post_cin:
            connection(cell_list[cin], cell_list[target_cin])
    
        post_ain = random.sample(all_ain_upper, int(dinr2ain_pC*len(all_ain_upper)))
        for ain in post_ain:
            connection(cell_list[cin], cell_list[ain])

        post_dinr = random.sample(all_dinr_upper, int(dinr2dinr_pC*len(all_dinr_upper)))
        for dinr in post_dinr:
            connection(cell_list[cin], cell_list[dinr])

    # indices_dinr = [i for i, x in enumerate(celltypes) if x == "dinr"]
    # indices_cin = [i for i, x in enumerate(celltypes) if x == "cin"]
    # indices_ain = [i for i, x in enumerate(celltypes) if x == "ain"]
    
    all_dinr_lower = left_index[indices_dinr[0]]
    all_dinr_upper = left_index[indices_dinr[1]]

    pre_dinr = all_dinr_lower
    for dnr in pre_dinr:
        post_dnr = random.sample(all_dinr_upper, int(lower2upper_pC*len(all_dinr_upper)))
        for target_dnr in post_dnr:
            # print("bird pre: ", dinr, "post: ", target_dinr)
            connection(cell_list[dnr], cell_list[target_dnr])

    all_dinr_lower = right_index[indices_dinr[0]]
    all_dinr_upper = right_index[indices_dinr[1]]
    print("all_dinr_lower: ", all_dinr_lower)
    print("all_dinr_upper: ", all_dinr_upper)
    pre_dinr = all_dinr_lower
    for dnr in pre_dinr:
        post_dnr = random.sample(all_dinr_upper, int(lower2upper_pC*len(all_dinr_upper)))
        for target_dnr in post_dnr:
            print("pre: ", dnr, "post: ", target_dnr)
            connection(cell_list[dnr], cell_list[target_dnr])

def setup_clamps(cell_list, sim_dur=5000):
    ind_cin = celltypes.index("cin")
    for i in left_index[ind_cin]:
        cell_list[i].set_clamps(duration=sim_dur)
    for i in right_index[ind_cin]:
        cell_list[i].set_clamps(duration=sim_dur)
        


def many_simulations( cin_wfac = cin_wfac, dep_gaba_tau = dep_gaba_tau):

    
    tstop = 5000
    sim_num = 10
    rate_spike_drive = 0.01


    for cin_wf in cin_wfac:
        for dep_gaba_t in dep_gaba_tau:
            brick_simulation(tstop, sim_num, rate_spike_drive, prb_ext_drive = prb_ext_drive, weight=8e-4, cin_wf=cin_wf, dep_gaba_t=dep_gaba_t)


def brick_simulation(tstop, sim_num, rate_spike_drive, prb_ext_drive = 0.1, weight=0.012e3, cin_wf=5000, dep_gaba_t=10):
    t_start = time.time()
    seed = 123#random.getrandbits(32)
    rnd.seed(seed)

    print("Running Simulation .... ")

    print("Create Cells")
    cell_list = CellCreation(RecAll=RecAll, sim_num=sim_num, cin_wf=cin_wf, dep_gaba_t=10)
    print("Cells Created.")

    print("Create connectivity .. ")
    CreateConfigAdj(cell_list)
    GapJunction(cell_list)
    print("Connectivity created.")

    print("Setting up external drive ..")
    ext_syn_drive(cell_list, rate_spike_drive=rate_spike_drive, sim_dur=5000, prb_ext_drive=prb_ext_drive, weight=weight)
    print("External drive set.")

    print("Setting up dc input..")
    setup_clamps(cell_list, sim_dur=5000)
    print("DC input set.")
    

    print("Running simulation .. ")
    RunSim(tstop=tstop, dt=dt)
    print("End of simulation. ")

    # save spike trains
    
    # out_idata = {i:cell.record["i_depAmpaNmda_soma"] for i,cell in enumerate(cell_list) if cell.whatami=="cin"}
    out_vdata = {i:cell.soma_volt for i,cell in enumerate(cell_list) if cell.whatami=="dinr"}
    out_vdata_cin = {i:cell.soma_volt for i,cell in enumerate(cell_list) if cell.whatami=="cin"}
    # out_vdata_ain = {i:cell.soma_volt for i,cell in enumerate(cell_list) if cell.whatami=="ain"}
    # out_stimt_ampa = {i:cell.stim_t_ampa for i,cell in enumerate(cell_list) if (cell.whatami=="dinr" and hasattr(cell, "stim_t_ampa")==1)}
    # out_stimid_ampa = {i:cell.stim_id_ampa for i,cell in enumerate(cell_list) if (cell.whatami=="dinr" and hasattr(cell, "stim_t_ampa")==1)}

    # out_stimt_nmda = {i:cell.stim_t_nmda for i,cell in enumerate(cell_list) if (cell.whatami=="dinr" and hasattr(cell, "stim_t_nmda")==1)}
    # out_stimid_nmda = {i:cell.stim_id_nmda for i,cell in enumerate(cell_list) if (cell.whatami=="dinr" and hasattr(cell, "stim_t_nmda")==1)}

   
    # with open('output_ifile.txt', 'wb') as fp:
    #     pickle.dump(out_idata, fp) # use `pickle.loads` to do the reverse
    
    with open('output_vfile_dinr_{:3.1f}gabaTau_{:6.2f}cinWf_noain.txt'.format(dep_gaba_t, cin_wf), 'wb') as fp:
        pickle.dump(out_vdata, fp) # use `pickle.loads` to do the reverse
    with open('output_vfile_cin_{:3.1f}gabaTau_{:6.2f}cinWf_noain.txt'.format(dep_gaba_t, cin_wf), 'wb') as fp:
        pickle.dump(out_vdata_cin, fp) # use `pickle.loads` to do the reverse
    # with open('output_vfile_ain.txt', 'wb') as fp:
    #     pickle.dump(out_vdata_ain, fp) # use `pickle.loads` to do the reverse
    # with open('output_stimt_stimid_ampa.txt', 'wb') as fp:
    #     pickle.dump({"stim_t":out_stimt_ampa, "stim_id":out_stimid_ampa}, fp)
    # with open('output_stimt_stimid_nmda.txt', 'wb') as fp:
    #     pickle.dump({"stim_t":out_stimt_nmda, "stim_id":out_stimid_nmda}, fp)
    # with open('output_nmda_dinr.txt', 'wb') as fp:
    #     pickle.dump(out_nmda_dinr, fp)
        
    t_end = time.time()
    print("Simulation Took {0}s.".format(t_end-t_start))


def RunSim(v_init=-80.0,tstop=0.0,dt=0.01):
    t_start = time.time()
    h.dt = dt
    h.t = 0.0
    counter=0
    h.finitialize(v_init)
    while h.t<tstop:
        counter += 1
        print("here: ", counter)
        h.fadvance()
        

many_simulations( cin_wfac = cin_wfac, dep_gaba_tau = dep_gaba_tau)

