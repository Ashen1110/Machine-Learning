import random
import numpy as np
import matplotlib.pyplot as plt

##### Q1
def random_generate(m,b,num_sample):
    # samples = (1,x,y)
    samples = np.zeros([num_sample, 3])
    samples[:,0]=1
    samples[:, 1] = np.random.uniform(0, 1000, num_sample)
    random_y = np.random.uniform(1, 1000, num_sample)
    random_y[int(num_sample/2):] = random_y[int(num_sample/2):]*-1
    samples[:, 2] = m * samples[:, 1] + b - random_y
    
    label = np.zeros([num_sample])
    label = np.ones((num_sample,), dtype=int)
    label[int(num_sample/2):] = -1
    
    return samples, label

#### get sign
def get_sign(w,x):
    return np.sign(np.sum(w*x))

#### verify whether the line equation is correct or not
def verification(samples, labels, w, num_sample, iteration, show = False):
    count_error = 0
    for i in range(num_sample):
        if get_sign(w, samples[i,:].reshape([1,3])) != labels[i]:
            count_error += 1
    if show == True:
        print(f"y = {-w[0, 1] / w[0, 2]:.5f} * x + {-w[0, 0] / w[0, 2]:.5f}")
        print(f"iteration = {iteration}")
        if count_error == 0:
            print("Line Correct!")
        else:
            print("Line Wrong!")
    return count_error/num_sample

#### Q2
def PLA(samples, labels, w,num_sample):
    iteration = 0
    while True:
        error_exist = False
        rand = random.sample(range(num_sample), num_sample)
        for i in rand:
            if get_sign(w, samples[i,:].reshape([1,3])) != labels[i]:
                w = w + labels[i]*(samples[i,:].reshape([1,3]))
                error_exist = True
                iteration +=1
                
        if error_exist == False:
            break
    return w, iteration


#### Q3
def Pocket(samples, labels, w,num_sample, threshold, con_threshold):
    iteration = 0
    counting = 0
    best_err_rate = 1
    
    while(iteration < threshold and counting < con_threshold):
        rand = random.sample(range(num_sample), num_sample)
        for i in rand:
            if get_sign(w, samples[i,:].reshape([1,3])) != labels[i]:
                w = w + labels[i]*(samples[i,:].reshape([1,3]))
                
                error_rate = verification(samples, labels, w,num_sample, iteration=None, show=False)
                if(error_rate < best_err_rate):
                    best_w= w
                    counting = 0
                    best_err_rate = error_rate
                else:
                    counting += 1
                    
                iteration += 1
                if best_err_rate == 0:
                    break
            #print(f"iteration = {iteration}, counting={counting}, rate of error = {best_err_rate}")
            
        if(best_err_rate == 0):
            break
    return best_w, iteration

#### Plot figure
class Plot():
    def __init__(self, samples, labels, title):
        plt.figure(figsize = (13,11))
        plt.title(title)
        plt.xlabel("x-axis")
        plt.ylabel("y-axis")
        
        positive = np.where(np.array(labels)==1)
        negative = np.where(np.array(labels)==-1)
        plt.plot(samples[positive, 1].squeeze().tolist(), samples[positive, 2].squeeze().tolist(),'o', label="positive samples on the right")
        plt.plot(samples[negative, 1].squeeze().tolist(), samples[negative, 2].squeeze().tolist(),'x', label="negative samples on the left")
        
    def plt_line(self, w, m, b, num_samples, iteration, label, content):
        if w is not None:
            m = -1 * w[0,1] / w[0,2]
            b = -1 * w[0,0] / w[0,2]
        line_x = np.arange(1000 + 1)
        line_y = m * line_x + b
        plt.plot(line_x, line_y, label=f"{label}: y = {m:.5f} * x + {b:.5f}")
        
    def save_show(self, itr_avg, filename, show_avg = False):
        if show_avg is True:
            print(f"Average iteration is = {itr_avg:.3f}\n")
            #plt.title(f"Average iteration is = {itr_avg:.3f}", loc = "right")
        plt.legend(loc="best")
        plt.savefig(filename, dpi=800, bbox_inches="tight")
        
        plt.show()