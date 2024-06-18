from rt605 import RT605
import os
import argparse
import numpy as np

def main():

    parser = argparse.ArgumentParser(description="RT605 simulation")

    parser.add_argument('--path_dir',default='XY_circle_path.txt',help='reference trajectory file name',type=str)
    parser.add_argument('--GTorMode',help='set gravity torque on/off',type=bool)
    parser.add_argument('--FricTorMode',help='set friction torque on/off',type=bool)
    
    #parser.add_argument('--gain',nargs='+',help='set apecific link gain(ex:kpp0-->set joint 0 kpp to value')
    parser.add_argument('-kpp1','--Kpp1', help='set joint 1 Kpp gain(ex: --Kpp1 100)', type=np.float32)
    parser.add_argument('-kpp2','--Kpp2', help='set joint 2 Kpp gain', type=np.float32)
    parser.add_argument('-kpp3','--Kpp3', help='set joint 3 Kpp gain', type=np.float32)
    parser.add_argument('-kpp4','--Kpp4', help='set joint 4 Kpp gain', type=np.float32)
    parser.add_argument('-kpp5','--Kpp5', help='set joint 5 Kpp gain', type=np.float32)
    parser.add_argument('-kpp6','--Kpp6', help='set joint 6 Kpp gain', type=np.float32)

    parser.add_argument('-kpi1','--Kpi1', help='set joint 1 Kpi gain', type=np.float32)
    parser.add_argument('-kpi2','--Kpi2', help='set joint 2 Kpi gain', type=np.float32)
    parser.add_argument('-kpi3','--Kpi3', help='set joint 3 Kpi gain', type=np.float32)
    parser.add_argument('-kpi4','--Kpi4', help='set joint 4 Kpi gain', type=np.float32)
    parser.add_argument('-kpi5','--Kpi5', help='set joint 5 Kpi gain', type=np.float32)
    parser.add_argument('-kpi6','--Kpi6', help='set joint 6 Kpi gain', type=np.float32)

    parser.add_argument('-kvp1','--Kvp1', help='set joint 1 Kvp gain(ex: --Kvp1 100)', type=np.float32)
    parser.add_argument('-kvp2','--Kvp2', help='set joint 2 Kvp gain', type=np.float32)
    parser.add_argument('-kvp3','--Kvp3', help='set joint 3 Kvp gain', type=np.float32)
    parser.add_argument('-kvp4','--Kvp4', help='set joint 4 Kvp gain', type=np.float32)
    parser.add_argument('-kvp5','--Kvp5', help='set joint 5 Kvp gain', type=np.float32)
    parser.add_argument('-kvp6','--Kvp6', help='set joint 6 Kvp gain', type=np.float32)

    parser.add_argument('-kvi1','--Kvi1', help='set joint 1 Kvi gain', type=np.float32)
    parser.add_argument('-kvi2','--Kvi2', help='set joint 2 Kvi gain', type=np.float32)
    parser.add_argument('-kvi3','--Kvi3', help='set joint 3 Kvi gain', type=np.float32)
    parser.add_argument('-kvi4','--Kvi4', help='set joint 4 Kvi gain', type=np.float32)
    parser.add_argument('-kvi5','--Kvi5', help='set joint 5 Kvi gain', type=np.float32)
    parser.add_argument('-kvi6','--Kvi6', help='set joint 6 Kvi gain', type=np.float32)

    parser.add_argument('-pf','--plotFreqMode', action='store_true', help='Enable plot frequency response of joints')
    parser.add_argument('-pc','--plotCartesianMode',action='store_true', help='Enable plot cartesian mode')
    parser.add_argument('-pj','--plotJointMode', action='store_true', help='Enable plot joints mode')
    parser.add_argument('-pp','--plotPolarMode',action='store_true', help='Enable plot polar mode')

    parser.add_argument('-s','--save',action='store_true', help='Enable save result')
    parser.add_argument('-name','--name',help='Name of this trial',type=str)

    opt = parser.parse_args()

    path_name = opt.path_dir
    GTorMode = opt.GTorMode
    FricTorMode = opt.FricTorMode
    trial_name = opt.name

    Kpp = []
    Kpi = []
    Kvp = []
    Kvi = []

    Kpp.append(opt.Kpp1)
    Kpp.append(opt.Kpp2)
    Kpp.append(opt.Kpp3)
    Kpp.append(opt.Kpp4)
    Kpp.append(opt.Kpp5)
    Kpp.append(opt.Kpp6)

    Kpi.append(opt.Kpi1)
    Kpi.append(opt.Kpi2)
    Kpi.append(opt.Kpi3)
    Kpi.append(opt.Kpi4)
    Kpi.append(opt.Kpi5)
    Kpi.append(opt.Kpi6)

    Kvp.append(opt.Kvp1)
    Kvp.append(opt.Kvp2)
    Kvp.append(opt.Kvp3)
    Kvp.append(opt.Kvp4)
    Kvp.append(opt.Kvp5)
    Kvp.append(opt.Kvp6)

    Kvi.append(opt.Kvi1)
    Kvi.append(opt.Kvi2)
    Kvi.append(opt.Kvi3)
    Kvi.append(opt.Kvi4)
    Kvi.append(opt.Kvi5)
    Kvi.append(opt.Kvi6)

    save_dir = f'./run/{trial_name}'

    if trial_name==None and opt.save:
        i=0
        while os.path.exists(f'./run/exp{i}'):
            i = i+1
        save_dir = f'./run/exp{i}'
        os.makedirs(save_dir)

    # save result (or not)
    if opt.save:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    # initialize RT605
    rt605 = RT605()
    rt605.initialize_model() # load servo inforamtion

    #  load trajectory (Initalize postion)
    path_file_dir = './data/Path/'
    if path_name!= None:
        rt605.load_HRSS_trajectory(path_file_dir+path_name)
    else:
        #path_name = 'XY_circle_path.txt'
        rt605.load_trajectory(path_file_dir+path_name) 
    
    if GTorMode !=None:
        rt605.compute_GTorque.enable_Gtor(GTorMode) #Ture on/off gravity

    if FricTorMode!=None:
        rt605.compute_friction.enable_friction(FricTorMode) #Ture on/off friction


    for i in range(6):
        if Kvi[i]!=None:
            rt605.setPID(i,gain="kvi",value=Kvi[i])
        if Kvp[i]!=None:
            rt605.setPID(i,gain="kvp",value=Kvp[i])
        if Kpi[i]!=None:
            rt605.setPID(i,gain="kpi",value=Kpi[i])
        if Kpp[i]!=None:
            rt605.setPID(i,gain="kpp",value=Kpp[i])

    # Start simulation
    rt605.start()
    

    if opt.save:
        rt605.save_log(save_dir)
        

    if opt.plotFreqMode:
        freq_fig = rt605.freq_response()   
        if opt.save:
            freq_fig.savefig(save_dir+'/bode.png')

    if opt.plotCartesianMode:
        cartesian_fig,cartesian_3D = rt605.plot_cartesian()
        if opt.save:
            cartesian_fig.savefig(save_dir+'/cartesian.png')
            cartesian_3D.savefig(save_dir+'/cartesian_3D.png')

    if opt.plotJointMode:
        joint_fig = rt605.plot_joint()
        if opt.save:
            joint_fig.savefig(save_dir+'/joint.png')

    if opt.plotPolarMode:
        polar_fig = rt605.plot_polar()
        if opt.save:
            polar_fig.savefig(save_dir+'/polar.png')




if __name__ == "__main__":
    main()




