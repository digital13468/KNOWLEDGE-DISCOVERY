import numpy as np
import pylab, math
import os
from scipy.cluster.vq import kmeans2
import scipy.cluster.hierarchy as hac

def read_data(filename):
    f = open(os.path.join(filename), 'r')
    data = np.genfromtxt(filename)
    f.close()
    return data
def add_subplot(fig):
    n = len(fig.axes)
    #print n
    for i in range(n):
        fig.axes[i].change_geometry(int(math.ceil((n+1)**0.5)), int(math.ceil((n+1)**0.5)), i+1)
    return n
def distance(p0, p1):
    #print p0
    #print p1
    return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)
def plot_assignment(data, old_seeds, fig, most_similar_object, color, iteration, start, end):
    n = add_subplot(fig)
    ax = fig.add_subplot(int(math.ceil((n+1)**0.5)),int(math.ceil((n+1)**0.5)),n+1, xlim=[start,end], ylim=[start,end], aspect='equal')
    #cluster_x = np.empty((0,))
    #cluster_y = np.empty((0,))
    for i in range(old_seeds.shape[0]):
        cluster_x = np.empty((0,))
        cluster_y = np.empty((0,))
        for j in range(most_similar_object.shape[0]):
            if (most_similar_object[j] == old_seeds[i]).all():
                cluster_x = np.append(cluster_x, data[j][1])
                cluster_y = np.append(cluster_y, data[j][2])
        ax.plot(cluster_x, cluster_y, 'o', color=color[i%len(color)], label='cluster '+str(i+1))
    center_x = np.empty((0,))
    center_y = np.empty((0,))               
    for i in range(len(old_seeds)):
        center_x = np.append(center_x, old_seeds[i][0])
        center_y = np.append(center_y, old_seeds[i][1])
    ax.plot(center_x, center_y, '*', color='r', label='center')
    ax.set_title('iteration '+str(iteration)+'-assigning instances')
    ax.legend(numpoints=1)
def assignment(data, old_seeds, fig, iteration, color, start, end):
    most_similar_object = np.zeros((data.shape[0], data.shape[1]-1))
    for i in range(data.shape[0]):
        #most_similar_object[i] = -1
        similarity = 999999999
        for j in range(len(old_seeds)):
            if similarity > distance(data[i,1:], old_seeds[j]):
                similarity = distance(data[i,1:], old_seeds[j])
                most_similar_object[i] = old_seeds[j]
    
    
    plot_assignment(data, old_seeds, fig, most_similar_object, color, iteration, start, end)
    return most_similar_object
def plot_cluster(fig, data, new_seeds, old_seeds, similar_object, iteration, start, end):
    n = add_subplot(fig)
    ax = fig.add_subplot(int(math.ceil((n+1)**0.5)),int(math.ceil((n+1)**0.5)),n+1, xlim=[start,end], ylim=[start,end], aspect='equal')
    for i in range(old_seeds.shape[0]):
        cluster_x = np.empty((0,))
        cluster_y = np.empty((0,))
        for j in range(data.shape[0]):
            if (similar_object[j] == old_seeds[i]).all():
                cluster_x = np.append(cluster_x, data[j][1])
                cluster_y = np.append(cluster_y, data[j][2])
        ax.plot(cluster_x, cluster_y, 'o', color=color[i%len(color)], label='cluster '+str(i+1))
    center_x = np.empty((0,))
    center_y = np.empty((0,))
    for i in range(len(new_seeds)):
        center_x = np.append(center_x, new_seeds[i][0])
        center_y = np.append(center_y, new_seeds[i][1])
    ax.plot(center_x, center_y, '*', color='r', label='center')
    ax.set_title('iteration '+str(iteration)+'-calculate new cluster centers')
    ax.legend(numpoints=1)
def mean(similar_object, old_seed, data):
    object_number = 0
    total = np.zeros(data.shape[1]-1)
    for i in range(similar_object.shape[0]):
        if (similar_object[i] == old_seed).all():
            object_number = object_number + 1
            total = total + data[i,1:]
    
    return total/object_number
    
def cluster_means(data, similar_object, k, old_seeds, fig, iteration, start, end):
    new_seeds = np.zeros((k, data.shape[1]-1))
    for i in range(k):
        new_seeds[i] = mean(similar_object, old_seeds[i], data)
    #print old_seeds
    #print new_seeds
    plot_cluster(fig, data, new_seeds, old_seeds, similar_object, iteration, start, end)
    return new_seeds
def k_means(data, k, initial_seeds, color):
    
    iteration = 0
    fig = pylab.figure('k='+str(k)+', initial seeds='+str(initial_seeds))
    lx = min(data[:,1])
    rx = max(data[:,1])
    ly = min(data[:,2])
    ry = max(data[:,2])
    if lx < ly:
        start = math.floor(lx)
    else:
        start = math.floor(ly)
    if rx < ry:
        end = math.ceil(ry)
    else:
        end = math.ceil(rx)
    ax = fig.add_subplot(111, xlim=[start,end], ylim=[start,end], aspect='equal')
    ax.plot(data[:,1], data[:,2], 'o', color='b', label='instance')
    
    
    
    
    center_x = np.empty((0,))
    center_y = np.empty((0,))
    for i in range(len(initial_seeds)):
        center_x = np.append(center_x, data[initial_seeds[i]-1,1])
        center_y = np.append(center_y, data[initial_seeds[i]-1,2])
        #print center_x, center_y
    ax.plot(center_x, center_y, '*', color='r', label='center')
    ax.set_title('iteration '+str(iteration))
    ax.legend(numpoints=1)
    old_seeds = np.zeros((k, data.shape[1]-1))
    for i in range(len(initial_seeds)):
        #print old_seeds[i]
        #print data[initial_seeds[i],1:]
        old_seeds[i] = data[initial_seeds[i]-1,1:]
    sol = kmeans2(np.delete(data,0,1),k=old_seeds,minit='matrix')
    print 'sol: %s' %(str(sol))
    stable = False
    while not stable:
        iteration = iteration + 1
        similar_object = assignment(data, old_seeds, fig, iteration, color, start, end)
        new_seeds = cluster_means(data, similar_object, k, old_seeds, fig, iteration, start, end)
        #print old_seeds
        #print new_seeds
        if (old_seeds == new_seeds).all():
            stable = True
            print 'the centers are %s' %(new_seeds)
        else:
            old_seeds = new_seeds
def cluster_distance(link, data, cluster1, cluster2):
    #print cluster1, cluster2
    single_dist = 999999999
    complete_dist = 0
    for i in range(len(cluster1)):
        for j in range(len(cluster2)):
            if link == 'single':
                if distance(data[cluster1[i], 1:], data[cluster2[j], 1:]) < single_dist:
                    single_dist = distance(data[cluster1[i], 1:], data[cluster2[j], 1:])
            elif link == 'complete':
                if distance(data[cluster1[i], 1:], data[cluster2[j], 1:]) > complete_dist:
                    complete_dist = distance(data[cluster1[i], 1:], data[cluster2[j], 1:])
    if link == 'single':
        return single_dist
    elif link == 'complete':
        return complete_dist
def merge_cluster(data, link, clusters):
    distance_matrix = np.zeros((len(clusters), len(clusters)))
    merger1_index = -1
    merger2_index = -1
    merger_distance = 999999999
    #print clusters
    for i in range(len(clusters)-1):
        for j in range(i+1,len(clusters)):
            #print clusters[i], clusters[j]
            distance_matrix[i][j] = cluster_distance(link, data, [tupl for tupl in clusters[i]], [tupl for tupl in clusters[j]])
            if merger_distance > distance_matrix[i][j]:
                merger_distance = distance_matrix[i][j]
                merger1_index = i
                merger2_index = j
    new_cluster = []
    #print merger1_index, merger2_index
    for i in range(len(clusters)):
        if i == merger1_index:
            new_cluster.append(tuple([tupl for tupl in clusters[merger1_index]])+tuple([tupl for tupl in clusters[merger2_index]]))
        elif i == merger2_index:
            continue
        else:
            new_cluster.append(clusters[i])
    #plot_agglomerative_clustering(data, new.cluster)
    return new_cluster
def plot_agglomerative_clustering(data, cluster, fig, color, pointstyle, iteration, start, end, link):
    if iteration == 0:
        ax = fig.add_subplot(111, xlim=[start,end], ylim=[start,end], aspect='equal')
    else:
        n = add_subplot(fig)
        ax = fig.add_subplot(int(math.ceil((n+1)**0.5)),int(math.ceil((n+1)**0.5)),n+1, xlim=[start,end], ylim=[start,end], aspect='equal')
    axisNum = -1
    for i in range(len(cluster)):
        cluster_x = np.empty((0,))
        cluster_y = np.empty((0,))
        axisNum += 1
        for j in range(len(cluster[i])):
            cluster_x = np.append(cluster_x, data[cluster[i],1])
            cluster_y = np.append(cluster_y, data[cluster[i],2])
        ax.plot(cluster_x, cluster_y, pointstyle[axisNum/len(color)], color=color[axisNum%len(color)])
    ax.set_title(link + '-link, ' + str(len(cluster)) + ' cluster(s)')
    #ax.legend(numpoints=1)
    
def agglomerative_clustering(data, color, pointstyle, link):
    fig = pylab.figure(link)
    clusters = []
    clusters.append(list(tuple([int(tupl)-1]) for tupl in data[:,0]))
    iteration = 0
    lx = min(data[:,1])
    rx = max(data[:,1])
    ly = min(data[:,2])
    ry = max(data[:,2])
    if lx < ly:
        start = math.floor(lx)
    else:
        start = math.floor(ly)
    if rx < ry:
        end = math.ceil(ry)
    else:
        end = math.ceil(rx)
    plot_agglomerative_clustering(data, clusters[-1], fig, color, pointstyle, iteration, start, end, link)
    
    
    #clusters.append(merge_cluster(data, link, clusters[-1]))
    #print clusters
    for i in range(data.shape[0]-1):
        iteration += 1
        clusters.append(merge_cluster(data, link, clusters[-1]))
        plot_agglomerative_clustering(data, clusters[-1], fig, color, pointstyle, iteration, start, end, link)
    print clusters
if __name__ == '__main__':
    color = ('b', 'g', 'c', 'k', 'r', 'w')
    pointstyle = ['o', '*', 'x', '.']
    filename = 'irises.txt'
    data = read_data(filename)
    
    k_means(data, k=2, initial_seeds=[9, 15], color=color)
    k_means(data, k=3, initial_seeds=[6, 9, 15], color=color)
    #pylab.show()
    agglomerative_clustering(data, color, pointstyle, link='single')
    agglomerative_clustering(data, color, pointstyle, link='complete')
    single_z = hac.linkage(data[:,1:], method = 'single')
    complete_z = hac.linkage(data[:,1:], method = 'complete')
    
    #x.fit()
    print single_z, complete_z
    pylab.show()
