import numpy as np
from matplotlib import pyplot as plt 
import os
from scipy.signal import find_peaks 
import pickle
import imageio

def rasterPlot(datadir, center = [125, -450, 125], uniform=True, figname='raster.png', orderBy='y', position='center', includerecs=True, dur=None):
    if not isinstance(datadir, list):
        files = os.listdir(datadir)
        mem_files = [file for file in files if (file.startswith(position + 'membrane'))]
    else:
        files = os.listdir(datadir[0])
        mem_files = [file for file in files if (file.startswith(position + 'membrane'))]
    raster = {}
    for file in mem_files:
        if not isinstance(datadir, list):
            with open(os.path.join(datadir,file), 'rb') as fileObj:
                data = pickle.load(fileObj)
        else:
            data = combineMemFiles(datadir, file)
        for v, pos, pop in zip(data[0], data[1], data[2]):
            if isinstance(v, list):
                pks, _ = find_peaks(v,0)
            else:
                pks, _ = find_peaks(v.as_numpy(), 0)
            if len(pks):
                if uniform:
                    r = ((pos[0]-center[0])**2 + (pos[1]-center[1])**2 + (pos[2]-center[2])**2)**(0.5)
                else:
                    r = (pos[0]**2 + pos[1]**2)**(0.5)
                if orderBy == 'y':
                    raster[pos[1]] = {'t': [data[3][ind] for ind in pks],
                        'pop' : pop}
                elif orderBy == 'x':
                    raster[pos[0]] = {'t': [data[3][ind] for ind in pks],
                        'pop' : pop}
                elif orderBy == 'z':
                    raster[pos[3]] = {'t': [data[3][ind] for ind in pks],
                        'pop' : pop}
                elif orderBy == 'r':
                    raster[r] = {'t': [data[3][ind] for ind in pks],
                        'pop' : pop}
                else:
                    raster[pos[1]] = {'t': [data[3][ind] for ind in pks],
                        'pop' : pop}
    if includerecs:
        if not isinstance(datadir, list):
            with open(datadir+'recs.pkl', 'rb') as fileObj:
                data = pickle.load(fileObj)
        else:
            data = combineRecs(datadir)
        for pos, v, pop in zip(data['pos'], data['v'], data['cell_type']):
            if isinstance(v, list):
                pks, _ = find_peaks(v,0)
            else:
                pks, _ = find_peaks(v.as_numpy(), 0)
            if len(pks):
                if orderBy == 'y':
                    raster[pos[1]] = {'t': [data['t'][ind] for ind in pks],
                        'pop' : pop}
                elif orderBy == 'x':
                    raster[pos[0]] = {'t': [data['t'][ind] for ind in pks],
                        'pop' : pop}
                elif orderBy == 'z':
                    raster[pos[3]] = {'t': [data['t'][ind] for ind in pks],
                        'pop' : pop}
                elif orderBy == 'r':
                    r = (pos[0]**2 + pos[1]**2 + pos[2]**2)**(0.5)
                    raster[r] = {'t': [data['t'][ind] for ind in pks],
                        'pop' : pop}
                else:
                    print("Invalid oderBy. Using y")
                    raster[pos[1]] = {'t' : [data['t'][ind] for ind in pks], 'pop' : pop}
    pops = np.array(['L2e', 'L2i', 'L4e', 'L4i', 'L5e', 'L5i', 'L6e', 'L6i'])
    cols = ['blue', 'red', 'yellow', 'purple', 'green', 'black', 'gray', 'orange']
    popCount = [0 for i in range(len(pops))]
    for key in raster.keys():
        popind = np.argwhere(pops==raster[key]['pop'])[0][0]
        popCount[popind] = popCount[popind] + 1
        c = cols[popind]
        if popCount[popind] == 1:
            plt.plot(np.divide(raster[key]['t'],1000), [key for i in range(len(raster[key]['t']))], '.', color=c, label=raster[key]['pop'])
        else:
            plt.plot(np.divide(raster[key]['t'],1000), [key for i in range(len(raster[key]['t']))], '.', color=c)
    if dur:
        plt.xlim(0,dur)
    plt.legend()
    plt.savefig(figname)

def allSpeciesMov(datadir, outpath, vmins, vmaxes, figname, condition='Perfused', dur=10, extent=None, includeSpks=False):
    """"Generates an mp4 video with heatmaps for K+, Cl-, Na+, and O2 overlaid with spiking data"""
    try:
        os.mkdir(outpath)
    except:
        pass
    plt.ioff()
    specs = ['k', 'cl', 'na', 'o2']
    k_files = [specs[0]+'_'+str(i)+'.npy' for i in range(int(dur*1000)) if (i%100)==0]    
    cl_files = [specs[1]+'_'+str(i)+'.npy' for i in range(int(dur*1000)) if (i%100)==0]    
    na_files = [specs[2]+'_'+str(i)+'.npy' for i in range(int(dur*1000)) if (i%100)==0]    
    o2_files = [specs[3]+'_'+str(i)+'.npy' for i in range(int(dur*1000)) if (i%100)==0]    
    for k_file, cl_file, na_file, o2_file in zip(k_files, cl_files, na_files, o2_files):
        t = int(k_file.split('.')[0].split('_')[1])
        ttl = 't = ' + str(float(k_file.split('.')[0].split('_')[1]) / 1000) + ' s'
        fig = plt.figure(figsize=(12,7.6))
        fig.text(.45, 0.825, ttl, fontsize=20)
        fig.text(0.45, 0.9, condition, fontsize=20)
        if includeSpks:
            posBySpkTime = xyOfSpikeTime(datadir)
            spkTimes = [key for key in posBySpkTime if (t-50 < key <= t+50)]
        ## K+ plot
        ax1 = fig.add_subplot(141)
        data = np.load(datadir+k_file)
        im = plt.imshow(np.transpose(data.mean(2)), vmin=vmins[0], vmax=vmaxes[0], interpolation='nearest', origin='lower', extent=extent)
        # im = plt.imshow(np.transpose(data.mean(2)), vmin=vmins[0], vmax=vmaxes[0], interpolation='nearest', origin='lower')
        plt.colorbar(im, fraction=0.046, pad=0.04)
        if includeSpks:
            if len(spkTimes):
                for spkTime in spkTimes:
                    plt.plot(posBySpkTime[spkTime]['x'], posBySpkTime[spkTime]['y'], 'w*')
        plt.title(r'[K$^+$]$_{ECS}$ ', fontsize=20)
        ## Cl plot
        ax2 = fig.add_subplot(142)
        data = np.load(datadir+cl_file)
        im = plt.imshow(np.transpose(data.mean(2)), vmin=vmins[1], vmax=vmaxes[1], interpolation='nearest', origin='lower', extent=extent)
        plt.colorbar(im, fraction=0.046, pad=0.04)
        if includeSpks:
            if len(spkTimes):
                for spkTime in spkTimes:
                    plt.plot(posBySpkTime[spkTime]['x'], posBySpkTime[spkTime]['y'], 'w*')
        plt.title(r'[Cl$^-$]$_{ECS}$ ', fontsize=20)
        ## Na plot
        ax3 = fig.add_subplot(143)
        data = np.load(datadir+na_file)
        im = plt.imshow(np.transpose(data.mean(2)), vmin=vmins[2], vmax=vmaxes[2], interpolation='nearest', origin='lower', extent=extent)
        plt.colorbar(im, fraction=0.046, pad=0.04)
        if includeSpks:
            if len(spkTimes):
                for spkTime in spkTimes:
                    plt.plot(posBySpkTime[spkTime]['x'], posBySpkTime[spkTime]['y'], 'w*')
        plt.title(r'[Na$^+$]$_{ECS}$ ', fontsize=20)
        ## O2 plot 
        ax4 = fig.add_subplot(144)
        data = np.load(datadir+o2_file)
        im = plt.imshow(np.transpose(data.mean(2)), vmin=vmins[3], vmax=vmaxes[3], interpolation='nearest', origin='lower', extent=extent)
        plt.colorbar(im, fraction=0.046, pad=0.04)
        if includeSpks:
            if len(spkTimes):
                for spkTime in spkTimes:
                    plt.plot(posBySpkTime[spkTime]['x'], posBySpkTime[spkTime]['y'], 'w*')
        plt.title(r'[O$_2$]$_{ECS}$ ', fontsize=20)
        plt.tight_layout()
        
        # plt.tight_layout()
        fig.savefig(outpath + k_file[:-4] + '.png')
        plt.close()
    times = []
    filenames = os.listdir(path=outpath)
    for file in filenames: 
        times.append(float(file.split('_')[-1].split('.')[0])) 
    inds = np.argsort(times)
    filenames_sort = [filenames[i] for i in inds]
    imagesc = []
    for filename in filenames_sort: 
        imagesc.append(imageio.imread(os.path.join(outpath,filename))) 
    imageio.mimsave(figname, imagesc)

def combineMemFiles(datadirs, file):
    # combine mem files from fragmented runs
    ## load first file 
    with open(os.path.join(datadirs[0], file), 'rb') as fileObj:
        data = pickle.load(fileObj)
    ## convert voltages to lists
    for ind, v in enumerate(data[0]):
        data[0][ind] = list(v)
    ## load the rest of them 
    for datadir in datadirs[1:]:
        with open(os.path.join(datadir, file), 'rb') as fileObj:
            data0 = pickle.load(fileObj)
        for ind, v in enumerate(data0[0]):
            data[0][ind].extend(list(v))
        data[1].extend(data0[1])
        data[2].extend(data0[2])
        data[3].extend(data0[3])
    return data 

def xyOfSpikeTime(datadir, position='center'):
    if isinstance(datadir, list):
        files = os.listdir(datadir[0])
    else:
        files = os.listdir(datadir)
    mem_files = [file for file in files if (file.startswith(position + 'membrane'))]
    posBySpkTime = {}
    for file in mem_files:
        if isinstance(datadir, list):
            data = combineMemFiles(datadir, file)
        else:
            with open(os.path.join(datadir,file), 'rb') as fileObj:
                data = pickle.load(fileObj)
        for v, pos in zip(data[0],data[1]):
            if isinstance(v, list):
                pks, _ = find_peaks(v, 0)
            else:
                pks, _ = find_peaks(v.as_numpy(), 0)
            if len(pks):                
                for ind in pks:
                    t = int(data[3][ind])
                    if t in posBySpkTime.keys():
                        posBySpkTime[t]['x'].append(pos[0])
                        posBySpkTime[t]['y'].append(pos[1])
                        posBySpkTime[t]['z'].append(pos[2])
                    else:
                        posBySpkTime[t] = {}
                        posBySpkTime[t]['x'] = [pos[0]]
                        posBySpkTime[t]['y'] = [pos[1]]
                        posBySpkTime[t]['z'] = [pos[2]]
    return posBySpkTime

def plotMemExamples(datadir, position='center', cells=None):
    fig, axs = plt.subplots(6,4, sharex=True)
    if not cells:
        cells = [[1, 3, 11, 37, 46, 54], 
                [24, 44, 95, 20, 42, 199],
                [5, 132, 232, 2, 101, 116], 
                [13, 26, 92, 47, 73, 238]]
    if isinstance(datadir, list):
        files = os.listdir(datadir[0])
    else:
        files = os.listdir(datadir)
    mem_files = [file for file in files if (file.startswith(position + 'membrane'))]
    count = 0
    for file in mem_files:
        if isinstance(datadir, list):
            data = combineMemFiles(datadir, file)
        else:
            with open(os.path.join(datadir,file), 'rb') as fileObj:
                data = pickle.load(fileObj)
        for v, pos, pop in zip(data[0], data[1], data[2]):
            for col, nums in enumerate(cells):
                if count in nums:
                    row = np.argwhere(np.array(nums) == count)[0][0]
                    axs[row][col].plot([i/40e3 for i in range(len(v))], v, 'k')
                    if isinstance(v, list):
                        spks, _ = find_peaks(v, height=0)
                    else:
                        spks, _ = find_peaks(v.as_numpy(), height=0)
                    freq = len(spks) / (len(v)/40e3)
                    axs[row][col].set_title(pop + ': ' + str(round(freq,2)) + ' Hz')
            count = count + 1

def plotMemV(datadir, position='center'):
    plt.ioff()
    if isinstance(datadir, list):
        try:
            os.mkdir(datadir[-1] + 'vmembs/')
        except:
            pass
        files = os.listdir(datadir[0])
    else:
        try:
            os.mkdir(datadir + 'vmembs/')
        except:
            pass
        files = os.listdir(datadir)
    count = 0
    mem_files = [file for file in files if (file.startswith(position + 'membrane'))]
    for file in mem_files:
        if isinstance(datadir, list):
            data = combineMemFiles(datadir, file)
        else:
            with open(os.path.join(datadir,file), 'rb') as fileObj:
                data = pickle.load(fileObj)
        for v, pos, pop in zip(data[0], data[1], data[2]):
            plt.figure()
            plt.plot(v, 'k')
            if isinstance(v, list):
                spks, _ = find_peaks(v, height=0)
            else:
                spks, _ = find_peaks(v.as_numpy(), height=0)
            freq = len(spks) / (len(v)/40e3)
            plt.title(pop + ': ' + str(round(freq,2)) + ' Hz')
            if isinstance(datadir, list):
                plt.savefig(datadir[-1] + 'vmembs/' + pop + '_' + str(count) + '.png')
            else:
                plt.savefig(datadir + 'vmembs/' + pop + '_' + str(count) + '.png')
            count = count + 1
            plt.close()

def compareKwaves(dirs, labels, legendTitle, colors=None, trimDict=None, sbplt=None, figname=None):
    """plots K+ wave trajectories from sims stored in list of folders dirs"""
    # plt.figure(figsize=(10,6))
    for dr, l, c in zip(dirs, labels, colors):
        times = []
        wave_pos = []
        if isinstance(dr, list):
            for d in dr:
                f = open(d + 'wave_progress.txt', 'r')
                for line in f.readlines():
                    times.append(float(line.split()[0]))
                    wave_pos.append(float(line.split()[-2]))
                f.close()
        else:
            f = open(dr + 'wave_progress.txt', 'r')
            for line in f.readlines():
                times.append(float(line.split()[0]))
                wave_pos.append(float(line.split()[-2]))
            f.close()
        if sbplt:
            plt.subplot(sbplt)
        if trimDict:
            if d in trimDict.keys():
                plt.plot(np.divide(times,1000)[:trimDict[d]], wave_pos[:trimDict[d]], label=l, linewidth=5, color=c)
            else:
                plt.plot(np.divide(times,1000), wave_pos, label=l, linewidth=5, color=c)
        else:
            if colors:
                plt.plot(np.divide(times,1000), wave_pos, label=l, linewidth=5, color=c)
            else:
                plt.plot(np.divide(times,1000), wave_pos, label=l, linewidth=5)
    # legend = plt.legend(title=legendTitle, fontsize=12)#, bbox_to_anchor=(-0.2, 1.05))
    legend = plt.legend(fontsize=12, loc='upper left')#, bbox_to_anchor=(-0.2, 1.05))
    # plt.setp(legend.get_title(), fontsize=14)
    plt.ylabel('K$^+$ Wave Position ($\mu$m)', fontsize=16)
    plt.xlabel('Time (s)', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    if figname:
        plt.savefig(figname)

def combineRecs(dirs, recNum=None):
    """tool for combining recordings from multiple recordings.  dirs is a list of
    directories with data.  intended for use with state saving/restoring"""
    # open first file 
    if recNum:
        filename = 'recs' + str(recNum) + '.pkl'
    else:
        filename = 'recs.pkl'

    with open(os.path.join(dirs[0],filename),'rb') as fileObj:
        data = pickle.load(fileObj)
    ## convert first file to all lists 
    for key in data.keys():
        if isinstance(data[key], list) and key != 'cell_type':
            for i in range(len(data[key])):
                try:
                    data[key][i] = list(data[key][i])
                except:
                    pass
        else:
            data[key] = list(data[key])
    # load each new data file
    for datadir in dirs[1:]:
        with open(os.path.join(datadir,filename), 'rb') as fileObj:
            new_data = pickle.load(fileObj)
        ## extend lists in original data with new data
        for key in new_data.keys():
            if isinstance(new_data[key], list):
                for i in range(len(new_data[key])):
                    try:
                        data[key][i].extend(list(new_data[key][i]))
                    except:
                        pass
            else:
                data[key].extend(list(new_data[key]))
    return data

def traceExamples(datadir, figname, iss=[0, 7, 15], recNum=None):
    """Function for plotting Vmemb, as well as ion and o2 concentration, for selected (iss) recorded
    neurons"""
    if recNum:
        filename = 'recs' + str(recNum) + '.pkl'
    else:
        filename = 'recs.pkl'
        
    if isinstance(datadir, list):
        data = combineRecs(datadir)
    else:
        with open(os.path.join(datadir,filename), 'rb') as fileObj:
            data = pickle.load(fileObj)
    # fig = plt.figure(figsize=(18,9))
    fig, axs = plt.subplots(2,4)
    fig.set_figheight(9)
    fig.set_figwidth(18)
    for i in iss:
        l = r'%s $\mu$m' % str(np.round((data['pos'][i][0] ** 2 + data['pos'][i][1] ** 2 + data['pos'][i][2] ** 2)**(0.5),1))
        axs[0][0].plot(np.divide(data['t'],1000), data['v'][i], label=l)
        axs[1][0].plot(np.divide(data['t'],1000), data['o2'][i])
        axs[0][1].plot(np.divide(data['t'],1000), data['ki'][i])
        axs[1][1].plot(np.divide(data['t'],1000), data['ko'][i])
        axs[0][2].plot(np.divide(data['t'],1000), data['nai'][i])
        axs[1][2].plot(np.divide(data['t'],1000), data['nao'][i])
        axs[0][3].plot(np.divide(data['t'],1000), data['cli'][i])
        axs[1][3].plot(np.divide(data['t'],1000), data['clo'][i])
    
    leg = axs[0][0].legend(title='Radial Position', fontsize=11, bbox_to_anchor=(-0.275, 1.05))
    plt.setp(leg.get_title(), fontsize=15)
    axs[0][0].set_ylabel('Membrane Potential (mV)', fontsize=16)
    plt.setp(axs[0][0].get_xticklabels(), fontsize=14)
    plt.setp(axs[0][0].get_yticklabels(), fontsize=14)
    axs[0][0].text(-0.15, 1.0, 'A)', transform=axs[0][0].transAxes,
        fontsize=16, fontweight='bold', va='top', ha='right')

    axs[1][0].set_ylabel(r'Extracellular [O$_{2}$] (mM)', fontsize=16)
    plt.setp(axs[1][0].get_xticklabels(), fontsize=14)
    plt.setp(axs[1][0].get_yticklabels(), fontsize=14)
    axs[1][0].text(-0.15, 1., 'E)', transform=axs[1][0].transAxes,
        fontsize=16, fontweight='bold', va='top', ha='right')
    
    axs[0][1].set_ylabel(r'Intracellular [K$^{+}$] (mM)', fontsize=16)
    plt.setp(axs[0][1].get_xticklabels(), fontsize=14)
    plt.setp(axs[0][1].get_yticklabels(), fontsize=14)
    axs[0][1].text(-0.15, 1., 'B)', transform=axs[0][1].transAxes,
        fontsize=18, fontweight='bold', va='top', ha='right')
    
    axs[1][1].set_ylabel(r'Extracellular [K$^{+}$] (mM)', fontsize=16)
    plt.setp(axs[1][1].get_xticklabels(), fontsize=14)
    plt.setp(axs[1][1].get_yticklabels(), fontsize=14)
    axs[1][1].text(-0.15, 1.0, 'F)', transform=axs[1][1].transAxes,
      fontsize=18, fontweight='bold', va='top', ha='right')
    
    axs[0][2].set_ylabel(r'Intracellular [Na$^{+}$] (mM)', fontsize=16)
    plt.setp(axs[0][2].get_xticklabels(), fontsize=14)
    plt.setp(axs[0][2].get_yticklabels(), fontsize=14)
    axs[0][2].text(-0.15, 1.0, 'C)', transform=axs[0][2].transAxes,
      fontsize=18, fontweight='bold', va='top', ha='right')

    axs[1][2].set_ylabel(r'Extracellular [Na$^{+}$] (mM)', fontsize=16)
    plt.setp(axs[1][2].get_xticklabels(), fontsize=14)
    plt.setp(axs[1][2].get_yticklabels(), fontsize=14)
    axs[1][2].text(-0.15, 1.0, 'G)', transform=axs[1][2].transAxes,
      fontsize=18, fontweight='bold', va='top', ha='right')
    
    axs[0][3].set_ylabel(r'Intracellular [Cl$^{-}$] (mM)', fontsize=16)
    plt.setp(axs[0][3].get_xticklabels(), fontsize=14)
    plt.setp(axs[0][3].get_yticklabels(), fontsize=14)
    axs[0][3].text(-0.15, 1.0, 'D)', transform=axs[0][3].transAxes,
      fontsize=18, fontweight='bold', va='top', ha='right')
    
    axs[1][3].set_ylabel(r'Extracellular [Cl$^{-}$] (mM)', fontsize=16)
    plt.setp(axs[1][3].get_xticklabels(), fontsize=14)
    plt.setp(axs[1][3].get_yticklabels(), fontsize=14)
    axs[1][3].text(-0.15, 1.0, 'H)', transform=axs[1][3].transAxes,
      fontsize=18, fontweight='bold', va='top', ha='right')

    fig.text(0.55, 0.01, 'Time (s)', fontsize=16)
    plt.tight_layout()
    plt.savefig(figname)
    plt.close()

def allTraces(datadir, figname, recNum=None):
    """Function for plotting Vmemb, as well as ion and o2 concentration, for all recorded neurons"""
    if recNum:
        filename = 'recs' + str(recNum) + '.pkl'
    else:
        filename = 'recs.pkl'
        
    if isinstance(datadir, list):
        data = combineRecs(datadir)
    else:
        with open(os.path.join(datadir,filename), 'rb') as fileObj:
            data = pickle.load(fileObj)

    try:
        os.mkdir(datadir + 'all_traces/')
    except:
        pass 
    
    for i in range(len(data['v'])):
        fig, axs = plt.subplots(2,4)
        fig.set_figheight(9)
        fig.set_figwidth(18)
        l = data['cell_type'][i]
        axs[0][0].plot(np.divide(data['t'],1000), data['v'][i], label=l)
        axs[1][0].plot(np.divide(data['t'],1000), data['o2'][i])
        axs[0][1].plot(np.divide(data['t'],1000), data['ki'][i])
        axs[1][1].plot(np.divide(data['t'],1000), data['ko'][i])
        axs[0][2].plot(np.divide(data['t'],1000), data['nai'][i])
        axs[1][2].plot(np.divide(data['t'],1000), data['nao'][i])
        axs[0][3].plot(np.divide(data['t'],1000), data['cli'][i])
        axs[1][3].plot(np.divide(data['t'],1000), data['clo'][i])
    
        leg = axs[0][0].legend(title='Population', fontsize=11, bbox_to_anchor=(-0.275, 1.05))
        plt.setp(leg.get_title(), fontsize=15)
        axs[0][0].set_ylabel('Membrane Potential (mV)', fontsize=16)
        plt.setp(axs[0][0].get_xticklabels(), fontsize=14)
        plt.setp(axs[0][0].get_yticklabels(), fontsize=14)
        axs[0][0].text(-0.15, 1.0, 'A)', transform=axs[0][0].transAxes,
            fontsize=16, fontweight='bold', va='top', ha='right')
        axs[0][0].set_ylim(-80,45)

        axs[1][0].set_ylabel(r'Extracellular [O$_{2}$] (mM)', fontsize=16)
        plt.setp(axs[1][0].get_xticklabels(), fontsize=14)
        plt.setp(axs[1][0].get_yticklabels(), fontsize=14)
        axs[1][0].text(-0.15, 1., 'E)', transform=axs[1][0].transAxes,
            fontsize=16, fontweight='bold', va='top', ha='right')
        axs[1][0].set_ylim(0.0, 0.045)
        
        axs[0][1].set_ylabel(r'Intracellular [K$^{+}$] (mM)', fontsize=16)
        plt.setp(axs[0][1].get_xticklabels(), fontsize=14)
        plt.setp(axs[0][1].get_yticklabels(), fontsize=14)
        axs[0][1].text(-0.15, 1., 'B)', transform=axs[0][1].transAxes,
            fontsize=18, fontweight='bold', va='top', ha='right')
        axs[0][1].set_ylim(60, 145)

        axs[1][1].set_ylabel(r'Extracellular [K$^{+}$] (mM)', fontsize=16)
        plt.setp(axs[1][1].get_xticklabels(), fontsize=14)
        plt.setp(axs[1][1].get_yticklabels(), fontsize=14)
        axs[1][1].text(-0.15, 1.0, 'F)', transform=axs[1][1].transAxes,
        fontsize=18, fontweight='bold', va='top', ha='right')
        axs[1][1].set_ylim(2.0, 42.0)

        axs[0][2].set_ylabel(r'Intracellular [Na$^{+}$] (mM)', fontsize=16)
        plt.setp(axs[0][2].get_xticklabels(), fontsize=14)
        plt.setp(axs[0][2].get_yticklabels(), fontsize=14)
        axs[0][2].text(-0.15, 1.0, 'C)', transform=axs[0][2].transAxes,
        fontsize=18, fontweight='bold', va='top', ha='right')
        axs[0][2].set_ylim(10, 100)

        axs[1][2].set_ylabel(r'Extracellular [Na$^{+}$] (mM)', fontsize=16)
        plt.setp(axs[1][2].get_xticklabels(), fontsize=14)
        plt.setp(axs[1][2].get_yticklabels(), fontsize=14)
        axs[1][2].text(-0.15, 1.0, 'G)', transform=axs[1][2].transAxes,
        fontsize=18, fontweight='bold', va='top', ha='right')
        axs[1][2].set_ylim(120, 150)
        
        axs[0][3].set_ylabel(r'Intracellular [Cl$^{-}$] (mM)', fontsize=16)
        plt.setp(axs[0][3].get_xticklabels(), fontsize=14)
        plt.setp(axs[0][3].get_yticklabels(), fontsize=14)
        axs[0][3].text(-0.15, 1.0, 'D)', transform=axs[0][3].transAxes,
        fontsize=18, fontweight='bold', va='top', ha='right')
        axs[0][3].set_ylim(5, 30)
        
        axs[1][3].set_ylabel(r'Extracellular [Cl$^{-}$] (mM)', fontsize=16)
        plt.setp(axs[1][3].get_xticklabels(), fontsize=14)
        plt.setp(axs[1][3].get_yticklabels(), fontsize=14)
        axs[1][3].text(-0.15, 1.0, 'H)', transform=axs[1][3].transAxes,
        fontsize=18, fontweight='bold', va='top', ha='right')
        axs[1][3].set_ylim(100, 135)

        fig.text(0.55, 0.01, 'Time (s)', fontsize=16)
        plt.tight_layout()
        plt.savefig(datadir + 'all_traces/' + l + '_' + str(i) + figname)
        plt.close()

if __name__ == '__main__':
    from cfgReduced import cfg 
    # rasterPlot('Data/test_templates/', center=[cfg.sizeX/2, -cfg.sizeY/2, cfg.sizeZ], figname='Data/test_templates/raster.png')
    # datadir = 'Data/fixedConn1e-6_p6RateInp_unbalancedk   _origSyns_2s/'
    datadir = ['test_dir/', 'test_dir_cont/']
    plt.ion()
    plt.show()
    # outpath = 'Data/scaleConnWeight1e-6_poissonInputs_2s/mov_files/'
    # figname = 'Data/scaleConnWeight1e-6_poissonInputs_2s/all_species.mp4'
    # vmins = [3.5, 100, 30, 0.1]
    # vmaxes = [40, 130, 140, 0.1]
    # extent = (0,242.0,-1470.0, 0.0)
    # allSpeciesMov(datadir, outpath, vmins, vmaxes, figname, dur=2, extent=extent, includeSpks=True)
    # compareKwaves([[cfg.restoredir, cfg.filename]], [cfg.ox], 'Condition', colors=['r'], figname=cfg.filename+'kwave.png')
    compareKwaves([cfg.filename], [cfg.ox], 'Condition', colors=['r'], figname=cfg.filename+'kwave.png')
    # plotMemV(datadir)

# v0.0 - analysis functions for raster plots, traces, etc.
# v0.1 - updating functions to handle fragmented sims 
